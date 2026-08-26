package com.bruxecho.recorder.domain

import android.content.Context
import com.bruxecho.recorder.session.SessionReader
import com.bruxecho.recorder.util.StorageInfo
import java.io.File
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

private const val SIMULATED_PROCESSING_DELAY_MS = 1200L
private const val DEMO_CLIP_COUNT = 4

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
 * this same interface). Two modes, both deterministic rather than
 * random-per-call:
 *
 * - Demo mode off: numbers are seeded from the real session id, so the
 *   same session shows the same result across app runs, and "no result"
 *   is a real signal (session missing or never finished), not noise.
 * - Demo mode on ([DemoModeState]): returns a curated, presentation-ready
 *   14-night dataset with bundled playable sample clips, independent of
 *   whatever is actually on disk — that's the entire point of demo mode.
 */
class MockDetectionRepository(context: Context) : DetectionRepository {

    private val appContext = context.applicationContext

    override suspend fun getNightResult(sessionId: String): NightResult? = withContext(Dispatchers.Default) {
        // Stands in for real analysis time, so the UI's "processing" state
        // is genuinely exercised on every call, not just faked for a screenshot.
        delay(SIMULATED_PROCESSING_DELAY_MS)

        if (DemoModeState.isEnabled) {
            return@withContext demoResultFor(sessionId)
        }

        val dir = File(StorageInfo.sessionsRootDir(appContext), sessionId)
        val parsed = SessionReader.readOne(dir) ?: return@withContext null
        val startedAtIso = parsed.startedAt ?: return@withContext null
        val endedAtIso = parsed.endedAt ?: return@withContext null // crashed/never finished — no result

        val startedAt = runCatching { OffsetDateTime.parse(startedAtIso).toInstant() }.getOrNull() ?: return@withContext null
        val endedAt = runCatching { OffsetDateTime.parse(endedAtIso).toInstant() }.getOrNull() ?: return@withContext null

        buildMockResult(sessionId, startedAt, endedAt, Random(sessionId.hashCode().toLong()))
    }

    override suspend fun getRecentNights(limit: Int): List<NightSummary> = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) {
            return@withContext demoSummaries().take(limit)
        }

        SessionReader.listSessions(StorageInfo.sessionsRootDir(appContext))
            .mapNotNull { entry ->
                val parsed = entry.parsed ?: return@mapNotNull null
                val startedAtIso = parsed.startedAt ?: return@mapNotNull null
                if (parsed.endedAt == null) return@mapNotNull null
                val startedAt = runCatching { OffsetDateTime.parse(startedAtIso).toInstant() }.getOrNull()
                    ?: return@mapNotNull null
                val date = startedAt.atZone(ZoneId.systemDefault()).toLocalDate()
                NightSummary(entry.sessionId, date, mockEventCount(Random(entry.sessionId.hashCode().toLong())))
            }
            .sortedByDescending { it.date }
            .take(limit)
    }

    /** Whether a specific real session exists but can't produce a result (crashed / never finished). */
    suspend fun isSessionIncomplete(sessionId: String): Boolean = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) return@withContext false
        val dir = File(StorageInfo.sessionsRootDir(appContext), sessionId)
        val parsed = SessionReader.readOne(dir)
        parsed == null || parsed.endedAt == null
    }

    /** The most recent usable session id, or null if there's nothing to show yet. */
    /**
     * The most recent session, complete or not. Deliberately does **not**
     * skip past an incomplete one to find an older working result — if last
     * night's recording crashed, that's exactly what the user opening
     * "results" wants to be told, not silently hidden behind an older night.
     */
    suspend fun latestSessionId(): String? = withContext(Dispatchers.Default) {
        if (DemoModeState.isEnabled) return@withContext demoSummaries().firstOrNull()?.sessionId
        SessionReader.listSessions(StorageInfo.sessionsRootDir(appContext)).firstOrNull()?.sessionId
    }

    private fun buildMockResult(sessionId: String, startedAt: Instant, endedAt: Instant, rng: Random): NightResult {
        val durationS = Duration.between(startedAt, endedAt).seconds.coerceAtLeast(1L).toFloat()
        val eventCount = mockEventCount(rng)
        val events = buildEvents(eventCount, startedAt, durationS, rng, clipPathFor = { null })
        return NightResult(
            sessionId = sessionId,
            startedAt = startedAt,
            endedAt = endedAt,
            grindEventCount = eventCount,
            eventsPerHour = eventCount / (durationS / 3600f),
            totalGrindDurationS = events.sumOf { it.durationS.toDouble() }.toFloat(),
            events = events,
            snoreRatio = if (rng.nextInt(100) < 20) null else rng.nextDouble(0.0, 0.6).toFloat(),
            // Derived from a synthetic 7-day average rather than independent
            // noise — otherwise a 0-event night could nonsensically show as
            // "up 4% from last week".
            comparisonPrevious7d = if (rng.nextInt(100) < 10) {
                null
            } else {
                val previous7dAvg = rng.nextDouble(2.0, 12.0)
                ((eventCount - previous7dAvg) / previous7dAvg).toFloat()
            },
        )
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

    private fun mockEventCount(rng: Random): Int {
        if (rng.nextInt(100) < 15) return 0 // a genuinely quiet night is normal, not an error — see NightResultScreen's "no events" state
        return rng.nextInt(3, 22)
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
