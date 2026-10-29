package com.bruxecho.recorder.domain

import android.content.Context
import com.bruxecho.recorder.data.SessionSync
import com.bruxecho.recorder.data.db.BruxEchoDatabase
import com.bruxecho.recorder.data.db.EventConfidence
import com.bruxecho.recorder.data.db.EventType
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

private const val SIMULATED_PROCESSING_DELAY_MS = 1200L
private const val DEMO_CLIP_COUNT = 4
private const val MIN_PRIOR_NIGHTS_FOR_COMPARISON = 3

// oldest -> newest, 30 nights: a noisier "before" baseline for the first 16,
// then the same 14-night improving tail — so toggling the trends screen's
// 14/30-night range actually shows a different chart, not an identical one.
private val DEMO_PATTERN = listOf(
    20, 18, 22, 17, 25, 19, 16, 21, 18, 23, 15, 20, 17, 19, 16, 18,
    14, 16, 12, 19, 28, 15, 11, 9, 10, 7, 6, 5, 4, 3,
)

/**
 * The only [DetectionRepository] implementation this sprint — no detection
 * model exists yet (that's Sprint 6's `RealDetectionRepository`, behind
 * this same interface). Two modes:
 *
 * - Demo mode off: reads from Room ([SessionSync] keeps it mirrored from
 *   session.json). Event numbers are still deterministic, seeded from the
 *   session id by [SessionSync] at import time, so a session's numbers
 *   don't change across app runs — same guarantee as before Room existed,
 *   just persisted instead of recomputed on every read.
 * - Demo mode on ([DemoModeState]): returns a curated, presentation-ready
 *   dataset with bundled playable sample clips, entirely in-memory —
 *   deliberately never touches Room, so a demo never depends on (or
 *   pollutes) whatever is actually on disk.
 */
class MockDetectionRepository(context: Context) : SessionAwareDetectionRepository {

    private val appContext = context.applicationContext
    private val db by lazy { BruxEchoDatabase.getInstance(appContext) }

    override suspend fun getNightResult(sessionId: String): NightResult? = withContext(Dispatchers.Default) {
        // Stands in for real analysis time, so the UI's "processing" state
        // is genuinely exercised on every call, not just faked for a screenshot.
        delay(SIMULATED_PROCESSING_DELAY_MS)

        if (DemoModeState.isEnabled) {
            return@withContext demoResultFor(sessionId)
        }

        SessionSync.sync(appContext)
        buildRoomResult(sessionId)
    }

    override suspend fun getRecentNights(limit: Int): List<NightSummary> = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) {
            return@withContext demoSummaries().take(limit)
        }

        SessionSync.sync(appContext)
        db.sessionDao().getRecent(limit).filter { it.isComplete }.map { session ->
            val grindCount = db.eventDao().getForSession(session.sessionId).count { it.type == EventType.GRIND }
            NightSummary(session.sessionId, session.startedAt.atZone(ZoneId.systemDefault()).toLocalDate(), grindCount)
        }
    }

    /** Whether a specific real session exists but can't produce a result (crashed / never finished). */
    override suspend fun isSessionIncomplete(sessionId: String): Boolean = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) return@withContext false
        SessionSync.sync(appContext)
        val session = db.sessionDao().getBySessionId(sessionId)
        session == null || !session.isComplete
    }

    /**
     * The most recent session, complete or not. Deliberately does **not**
     * skip past an incomplete one to find an older working result — if last
     * night's recording crashed, that's exactly what the user opening
     * "results" wants to be told, not silently hidden behind an older night.
     */
    override suspend fun latestSessionId(): String? = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) return@withContext demoSummaries().firstOrNull()?.sessionId
        SessionSync.sync(appContext)
        db.sessionDao().getLatest()?.sessionId
    }

    private suspend fun buildRoomResult(sessionId: String): NightResult? {
        val session = db.sessionDao().getBySessionId(sessionId) ?: return null
        val endedAt = session.endedAt
        if (!session.isComplete || endedAt == null) return null

        val events = db.eventDao().getForSession(sessionId)
        val durationS = Duration.between(session.startedAt, endedAt).seconds.coerceAtLeast(1L).toFloat()

        val grindEvents = events.filter { it.type == EventType.GRIND }.map { e ->
            GrindEvent(
                id = "evt_${e.id}",
                offsetS = e.startS,
                timestamp = session.startedAt.plusSeconds(e.startS.toLong()),
                durationS = e.durationS,
                confidence = e.confidence.toDomain(),
                clipPath = e.clipPath,
            )
        }
        val snoreEvents = events.filter { it.type == EventType.SNORE }
        val snoreRatio = (snoreEvents.sumOf { it.durationS.toDouble() } / durationS).toFloat().coerceIn(0f, 1f)

        val date = session.startedAt.atZone(ZoneId.systemDefault()).toLocalDate()
        val priorNights = db.dailySummaryDao().getRange(date.minusDays(7), date.minusDays(1)).filter { it.isComplete }
        val comparison = if (priorNights.size < MIN_PRIOR_NIGHTS_FOR_COMPARISON) {
            null
        } else {
            val avg = priorNights.map { it.mildCount + it.severeCount }.average()
            if (avg <= 0.0) null else ((grindEvents.size - avg) / avg).toFloat()
        }

        return NightResult(
            sessionId = sessionId,
            startedAt = session.startedAt,
            endedAt = endedAt,
            grindEventCount = grindEvents.size,
            eventsPerHour = grindEvents.size / (durationS / 3600f),
            totalGrindDurationS = grindEvents.sumOf { it.durationS.toDouble() }.toFloat(),
            events = grindEvents,
            snoreRatio = snoreRatio,
            comparisonPrevious7d = comparison,
        )
    }

    private fun EventConfidence.toDomain(): Confidence = when (this) {
        EventConfidence.HIGH -> Confidence.HIGH
        EventConfidence.MEDIUM -> Confidence.MEDIUM
        EventConfidence.LOW -> Confidence.LOW
    }

    private fun buildEvents(
        count: Int,
        startedAt: Instant,
        durationS: Float,
        rng: Random,
        clipPathFor: (Int) -> String?,
    ): List<GrindEvent> {
        val offsets = List(count) { rng.nextDouble(0.0, durationS.toDouble()).toFloat() }.sorted()
        return offsets.mapIndexed { i, offsetS ->
            val confidenceRoll = rng.nextInt(100)
            GrindEvent(
                id = "evt_$i",
                offsetS = offsetS,
                timestamp = startedAt.plusSeconds(offsetS.toLong()),
                durationS = rng.nextDouble(0.4, 3.0).toFloat(),
                confidence = when {
                    confidenceRoll < 50 -> Confidence.HIGH
                    confidenceRoll < 80 -> Confidence.MEDIUM
                    else -> Confidence.LOW
                },
                clipPath = clipPathFor(i),
            )
        }
    }

    // --- Demo dataset: fixed, not time-random, so a demo looks the same every time it's opened. ---

    private fun demoDates(): List<LocalDate> {
        val today = LocalDate.now()
        return (DEMO_PATTERN.indices).map { i -> today.minusDays((DEMO_PATTERN.size - i).toLong()) }
    }

    private fun demoSessionId(index: Int) = "demo_night_$index"

    private fun demoSummaries(): List<NightSummary> {
        val dates = demoDates()
        return DEMO_PATTERN.indices.reversed().map { i ->
            NightSummary(demoSessionId(i), dates[i], DEMO_PATTERN[i])
        }
    }

    private fun demoResultFor(sessionId: String): NightResult {
        val index = sessionId.removePrefix("demo_night_").toIntOrNull()?.takeIf { it in DEMO_PATTERN.indices }
            ?: (DEMO_PATTERN.size - 1) // unknown/real id while demo is on -> fall back to the latest demo night
        val date = demoDates()[index]
        val startedAt = date.atTime(23, 15).atZone(ZoneId.systemDefault()).toInstant()
        val durationS = (7 * 3600 + 40 * 60).toFloat() // ~7h40m, a plausible night
        val endedAt = startedAt.plusSeconds(durationS.toLong())
        val eventCount = DEMO_PATTERN[index]
        val rng = Random(index * 97L + 13L) // fixed per-index seed, not the id's hashCode — keeps demo nights stable even if the id scheme ever changes
        var clipCursor = 0
        val events = buildEvents(eventCount, startedAt, durationS, rng) {
            val path = "asset:///demo_clips/grind_0${(clipCursor % DEMO_CLIP_COUNT) + 1}.wav"
            clipCursor++
            path
        }
        val previous7 = DEMO_PATTERN.subList(maxOf(0, index - 7), index)
        val comparison = if (previous7.isEmpty()) null else (eventCount - previous7.average()) / previous7.average()
        return NightResult(
            sessionId = demoSessionId(index),
            startedAt = startedAt,
            endedAt = endedAt,
            grindEventCount = eventCount,
            eventsPerHour = eventCount / (durationS / 3600f),
            totalGrindDurationS = events.sumOf { it.durationS.toDouble() }.toFloat(),
            events = events,
            snoreRatio = 0.12f + (index % 5) * 0.05f,
            comparisonPrevious7d = comparison?.toFloat(),
        )
    }
}
