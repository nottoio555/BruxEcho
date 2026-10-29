package com.bruxecho.recorder.data

import android.content.Context
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.DailySummaryEntity
import com.bruxecho.recorder.data.db.EventConfidence
import com.bruxecho.recorder.data.db.EventEntity
import com.bruxecho.recorder.data.db.EventType
import com.bruxecho.recorder.data.db.Severity
import com.bruxecho.recorder.data.db.SessionEntity
import com.bruxecho.recorder.session.SessionReader
import com.bruxecho.recorder.util.StorageInfo
import java.io.File
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

private val SESSION_ID_FORMATTER = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")
private const val SUPPORTED_EVENTS_SCHEMA = 1
private const val SEVERE_EVENT_S = 2.0f

private fun JSONObject.optStringOrNull(key: String): String? =
    if (!has(key) || isNull(key)) null else optString(key).takeIf { it.isNotBlank() }

/**
 * Keeps Room in sync with session.json — the on-disk files under
 * StorageInfo.sessionsRootDir() stay the source of truth (never deleted,
 * never modified here); this only mirrors their metadata into Room so the
 * calendar/stats screens can query 30-90 days at once instead of re-parsing
 * every session.json on every frame.
 *
 * Idempotent and safe to call on every app start: sessions already in Room
 * are skipped by id, so this naturally covers both the one-time historic
 * import (Sprint 4.5) and syncing newly-recorded sessions going forward,
 * without RecordingService needing to know Room exists.
 *
 * No detection model exists yet (see CLAUDE.md) — events for a newly-synced
 * *complete* session are deterministically generated mock data, seeded from
 * the session id exactly as [com.bruxecho.recorder.domain.MockDetectionRepository]
 * generated them before this table existed, so a session's numbers don't
 * change just because it moved from being recomputed-on-read to
 * stored-in-Room. Demo Mode data is intentionally NOT written here — it
 * stays synthetic/in-memory so it never depends on (or pollutes) real
 * on-device state.
 */
object SessionSync {

    suspend fun sync(context: Context) = withContext(Dispatchers.IO) {
        val db = BruxEchoDatabase.getInstance(context)
        val sessionDao = db.sessionDao()
        val eventDao = db.eventDao()
        val summaryDao = db.dailySummaryDao()

        // A session already synced as *complete* is immutable — session.json
        // never changes after ended_at is written — so it's safe to skip.
        // One synced as *incomplete* must be re-checked every call: the
        // recording may have since been stopped properly, and until now
        // that transition was never picked up (a session synced once while
        // still running stayed "incomplete" in Room forever, even after
        // finishing normally).
        val alreadyComplete = sessionDao.getCompleteIds().toSet()
        val entries = SessionReader.listSessions(StorageInfo.sessionsRootDir(context))

        for (entry in entries) {
            // A complete session's session.json never changes, so it can be
            // skipped — but its events.json can: analysis gets re-run with a
            // different threshold and pushed again. So anything analyzed is
            // always re-read. That costs one JSON parse per analyzed night per
            // sync, which is nothing next to showing stale counts forever.
            val analyzed = File(entry.dir, "events.json").isFile
            if (entry.sessionId in alreadyComplete && !analyzed) continue
            val parsed = entry.parsed

            val startedAt = parsed?.startedAt
                ?.let { runCatching { OffsetDateTime.parse(it).toInstant() }.getOrNull() }
                ?: parseInstantFromSessionId(entry.sessionId)
                ?: continue // no usable timestamp anywhere — nothing to place on the calendar

            val endedAt = parsed?.endedAt?.let { runCatching { OffsetDateTime.parse(it).toInstant() }.getOrNull() }
            val isComplete = parsed != null && endedAt != null

            sessionDao.upsert(
                SessionEntity(
                    sessionId = entry.sessionId,
                    startedAt = startedAt,
                    endedAt = endedAt,
                    audioSource = parsed?.audioSourceUsed,
                    sampleRate = parsed?.sampleRateActual,
                    segmentCount = parsed?.segments?.size ?: 0,
                    totalDurationS = parsed?.totalDurationS ?: 0.0,
                    isComplete = isComplete,
                    // Was hardcoded "app" on the assumption that everything
                    // under sessionsRootDir was recorded here. That stopped
                    // being true once analyzed external and synthetic-mixed
                    // sessions started being pushed to the device, and their
                    // numbers are not comparable to the app's own recordings.
                    provenance = parsed?.provenance ?: "app",
                ),
            )

            if (!isComplete) continue

            val durationS = (endedAt!!.epochSecond - startedAt.epochSecond).coerceAtLeast(1L).toFloat()
            // Real model output when this night has been analyzed, mock only as
            // a fallback. Without this the calendar showed placeholder counts
            // for a night whose results screen showed the real ones — one
            // session read "0 ครั้ง" on the calendar and 83 events on the
            // results screen, which is the same "two screens disagree" bug as
            // the duplicate-date one, one level down.
            val events = readAnalyzedEvents(entry.dir, entry.sessionId)
                ?: generateMockEvents(
                    entry.sessionId,
                    startedAt,
                    durationS,
                    Random(entry.sessionId.hashCode().toLong()),
                )
            // Idempotent even if this session was already synced once while
            // still incomplete (no events would exist yet then, but clearing
            // first makes this safe regardless of call order).
            eventDao.deleteForSession(entry.sessionId)
            eventDao.insertAll(events)
        }

        // Rebuild every day's totals rather than writing one row per session
        // as each is imported. Two things make this the right shape: a day can
        // hold more than one session (the per-session write silently dropped
        // all but the last), and rebuilding unconditionally self-heals rows
        // left wrong by the previous version, which the per-session loop never
        // would — it skips sessions that are already synced.
        rebuildDailySummaries(db)
    }

    /** Recompute `daily_summary` from whatever is currently in `sessions`/`events`. */
    private suspend fun rebuildDailySummaries(db: BruxEchoDatabase) {
        val zone = ZoneId.systemDefault()
        val sessions = db.sessionDao().getRecent(Int.MAX_VALUE)
        val byDate = sessions.groupBy { it.startedAt.atZone(zone).toLocalDate() }

        for ((date, daySessions) in byDate) {
            val events = daySessions.flatMap { db.eventDao().getForSession(it.sessionId) }
            val complete = daySessions.filter { it.isComplete }
            db.dailySummaryDao().upsert(
                DailySummaryEntity(
                    date = date,
                    // The day's newest session: what "open results" defaults to.
                    sessionId = daySessions.maxByOrNull { it.startedAt }!!.sessionId,
                    sleepDurationS = complete.sumOf { session ->
                        session.endedAt?.let { it.epochSecond - session.startedAt.epochSecond }?.toDouble() ?: 0.0
                    },
                    mildCount = events.count { it.type == EventType.GRIND && it.severity == Severity.MILD },
                    severeCount = events.count { it.type == EventType.GRIND && it.severity == Severity.SEVERE },
                    snoreLightCount = events.count { it.type == EventType.SNORE && it.severity == Severity.MILD },
                    snoreLoudCount = events.count { it.type == EventType.SNORE && it.severity == Severity.SEVERE },
                    // A day counts as complete once any session on it finished,
                    // so one crashed recording doesn't grey out a good night.
                    isComplete = complete.isNotEmpty(),
                ),
            )
        }
    }

    /**
     * Events from `events.json` if this night has been analyzed, else null.
     *
     * Reads the same file [com.bruxecho.recorder.domain.RealDetectionRepository]
     * does, so the calendar and the results screen cannot report different
     * numbers for one night.
     */
    private fun readAnalyzedEvents(dir: File, sessionId: String): List<EventEntity>? {
        val file = File(dir, "events.json")
        if (!file.isFile) return null
        return runCatching {
            val json = JSONObject(file.readText())
            if (json.optInt("schema_version", 1) > SUPPORTED_EVENTS_SCHEMA) return null
            val array = json.optJSONArray("events") ?: return null
            val events = mutableListOf<EventEntity>()
            for (i in 0 until array.length()) {
                val row = array.getJSONObject(i)
                val type = when (row.optString("label")) {
                    "bruxism" -> EventType.GRIND
                    "snore" -> EventType.SNORE
                    else -> continue
                }
                val startS = row.optDouble("start_s", 0.0).toFloat()
                val endS = row.optDouble("end_s", 0.0).toFloat()
                events += EventEntity(
                    sessionId = sessionId,
                    startS = startS,
                    endS = endS,
                    type = type,
                    // Duration, not confidence: "severe" on the calendar means a
                    // longer grinding episode, which is the clinically
                    // meaningful axis. Confidence is how sure the model is,
                    // which is a different question and already has its own
                    // display on the results screen.
                    severity = if (endS - startS >= SEVERE_EVENT_S) Severity.SEVERE else Severity.MILD,
                    confidence = when (row.optString("confidence_level").uppercase()) {
                        "HIGH" -> EventConfidence.HIGH
                        "LOW" -> EventConfidence.LOW
                        else -> EventConfidence.MEDIUM
                    },
                    clipPath = row.optStringOrNull("clip_file")?.let { File(dir, it).absolutePath },
                )
            }
            events
        }.getOrNull()
    }

    /** Mirrors [com.bruxecho.recorder.session.SessionReader.deleteSession] on the Room side — call both together. */
    suspend fun deleteSession(context: Context, sessionId: String) = withContext(Dispatchers.IO) {
        val db = BruxEchoDatabase.getInstance(context)
        db.eventDao().deleteForSession(sessionId)
        db.sessionDao().deleteBySessionId(sessionId)
        // Drop the row first so a date whose last session just went away
        // disappears from the calendar — the rebuild below only visits dates
        // that still have sessions. For a date that keeps other sessions the
        // rebuild puts a corrected row straight back.
        db.dailySummaryDao().deleteBySessionId(sessionId)
        rebuildDailySummaries(db)
    }

    private fun parseInstantFromSessionId(sessionId: String): Instant? =
        runCatching {
            java.time.LocalDateTime.parse(sessionId, SESSION_ID_FORMATTER)
                .atZone(ZoneId.systemDefault())
                .toInstant()
        }.getOrNull()

    /** Same distribution [com.bruxecho.recorder.domain.MockDetectionRepository] used pre-Room, plus a severity roll. */
    private fun generateMockEvents(sessionId: String, startedAt: Instant, durationS: Float, rng: Random): List<EventEntity> {
        val events = mutableListOf<EventEntity>()

        val grindCount = if (rng.nextInt(100) < 15) 0 else rng.nextInt(3, 22)
        val grindOffsets = List(grindCount) { rng.nextDouble(0.0, durationS.toDouble()).toFloat() }.sorted()
        grindOffsets.forEach { offsetS ->
            val confidenceRoll = rng.nextInt(100)
            val severityRoll = rng.nextInt(100)
            val eventDurationS = rng.nextDouble(0.4, 3.0).toFloat()
            events += EventEntity(
                sessionId = sessionId,
                startS = offsetS,
                endS = offsetS + eventDurationS,
                type = EventType.GRIND,
                severity = if (severityRoll < 20) Severity.SEVERE else Severity.MILD,
                confidence = when {
                    confidenceRoll < 50 -> EventConfidence.HIGH
                    confidenceRoll < 80 -> EventConfidence.MEDIUM
                    else -> EventConfidence.LOW
                },
                clipPath = null,
            )
        }

        // 80% of nights have some snoring, matching the old snoreRatio roll —
        // spread across a handful of discrete events instead of one float ratio.
        if (rng.nextInt(100) >= 20) {
            val snoreEventCount = rng.nextInt(1, 6)
            repeat(snoreEventCount) {
                val offsetS = rng.nextDouble(0.0, durationS.toDouble()).toFloat()
                val eventDurationS = rng.nextDouble(20.0, 180.0).toFloat().coerceAtMost(durationS - offsetS)
                if (eventDurationS <= 0f) return@repeat
                events += EventEntity(
                    sessionId = sessionId,
                    startS = offsetS,
                    endS = offsetS + eventDurationS,
                    type = EventType.SNORE,
                    severity = if (rng.nextInt(100) < 25) Severity.SEVERE else Severity.MILD,
                    confidence = EventConfidence.MEDIUM,
                    clipPath = null,
                )
            }
        }

        return events
    }
}
