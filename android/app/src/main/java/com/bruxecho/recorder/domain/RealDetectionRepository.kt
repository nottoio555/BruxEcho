package com.bruxecho.recorder.domain

import android.content.Context
import com.bruxecho.recorder.util.StorageInfo
import java.io.File
import java.time.Duration
import java.time.Instant
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

private const val EVENTS_FILE = "events.json"
private const val SUPPORTED_SCHEMA_VERSION = 1
private const val LABEL_BRUXISM = "bruxism"
private const val COMPARISON_WINDOW_NIGHTS = 7
private const val MIN_PRIOR_NIGHTS_FOR_COMPARISON = 3

/**
 * Real detection results, read from the `events.json` that
 * `tools/process_session.py` writes into a session folder.
 *
 * The desktop pipeline does the analysis (YAMNet + the trained head is far
 * too heavy to run on every phone), so this class is deliberately only a
 * reader: no model, no inference, no scoring. A session that hasn't been
 * analyzed yet simply has no events.json, and [hasResults] returns false so
 * [DetectionRepositoryProvider] can fall back.
 *
 * Clip paths are resolved to absolute filesystem paths here, which is the
 * shape `ClipPlayer` already understands for non-asset clips.
 */
class RealDetectionRepository(context: Context) : SessionAwareDetectionRepository {

    private val appContext = context.applicationContext

    private fun sessionDir(sessionId: String) = File(StorageInfo.sessionsRootDir(appContext), sessionId)

    private fun eventsFile(sessionId: String) = File(sessionDir(sessionId), EVENTS_FILE)

    /**
     * Whether this session has *usable* analysis output.
     *
     * Deliberately parses rather than just checking the file exists: the
     * provider uses this to decide which repository answers, and a present
     * but unreadable events.json would otherwise route the screen here to be
     * told "no result" instead of falling back to the placeholder numbers.
     */
    fun hasResults(sessionId: String): Boolean = readEvents(sessionId) != null

    override suspend fun getNightResult(sessionId: String): NightResult? = withContext(Dispatchers.IO) {
        val parsed = readEvents(sessionId) ?: return@withContext null
        val priorCounts = priorNightCounts(sessionId, parsed.startedAt)
        parsed.toNightResult(comparison = comparisonAgainst(parsed.grindEvents.size, priorCounts))
    }

    override suspend fun getRecentNights(limit: Int): List<NightSummary> = withContext(Dispatchers.IO) {
        allAnalyzedNights()
            .sortedByDescending { it.startedAt }
            .take(limit)
            .map {
                NightSummary(
                    sessionId = it.sessionId,
                    date = it.startedAt.atZone(ZoneId.systemDefault()).toLocalDate(),
                    grindEventCount = it.grindEvents.size,
                )
            }
    }

    override suspend fun isSessionIncomplete(sessionId: String): Boolean = withContext(Dispatchers.IO) {
        // Analyzed results settle the question: a night we have events for
        // plainly finished. This check comes first because `--push` of an
        // externally-imported night lands events.json and the clips without a
        // session.json, and the file check below would then call a night with
        // real results "incomplete" and hide them behind a crash message.
        if (readEvents(sessionId) != null) return@withContext false

        // Otherwise "incomplete" means the recording itself never finished.
        val dir = sessionDir(sessionId)
        if (!dir.isDirectory) return@withContext true
        val meta = File(dir, "session.json")
        if (!meta.isFile) return@withContext true
        runCatching { JSONObject(meta.readText()).isNull("ended_at") }.getOrDefault(true)
    }

    override suspend fun latestSessionId(): String? = withContext(Dispatchers.IO) {
        allAnalyzedNights().maxByOrNull { it.startedAt }?.sessionId
    }

    private fun allAnalyzedNights(): List<ParsedNight> {
        val root = StorageInfo.sessionsRootDir(appContext)
        val dirs = root.listFiles { file -> file.isDirectory } ?: return emptyList()
        return dirs.mapNotNull { readEvents(it.name) }
    }

    private fun priorNightCounts(sessionId: String, startedAt: Instant): List<Int> =
        allAnalyzedNights()
            .filter { it.sessionId != sessionId && it.startedAt.isBefore(startedAt) }
            .sortedByDescending { it.startedAt }
            .take(COMPARISON_WINDOW_NIGHTS)
            .map { it.grindEvents.size }

    private fun comparisonAgainst(count: Int, priorCounts: List<Int>): Float? {
        if (priorCounts.size < MIN_PRIOR_NIGHTS_FOR_COMPARISON) return null
        val average = priorCounts.average()
        if (average <= 0.0) return null
        return ((count - average) / average).toFloat()
    }

    private fun readEvents(sessionId: String): ParsedNight? {
        val file = eventsFile(sessionId)
        if (!file.isFile) return null
        return runCatching {
            val json = JSONObject(file.readText())
            val version = json.optInt("schema_version", 1)
            // Refuse a future schema rather than silently misreading it: a
            // wrong count shown as fact is worse than showing nothing.
            if (version > SUPPORTED_SCHEMA_VERSION) return@runCatching null

            val startedAt = json.optStringOrNull("started_at")?.toInstantOrNull() ?: return@runCatching null
            val endedAt = json.optStringOrNull("ended_at")?.toInstantOrNull()
                ?: startedAt.plusSeconds(json.optDouble("total_duration_s", 0.0).toLong())
            val summary = json.optJSONObject("summary")
            val clipRoot = sessionDir(sessionId)

            val grindEvents = mutableListOf<GrindEvent>()
            val array = json.optJSONArray("events")
            if (array != null) {
                for (i in 0 until array.length()) {
                    val row = array.getJSONObject(i)
                    if (row.optString("label") != LABEL_BRUXISM) continue
                    val startS = row.optDouble("start_s", 0.0).toFloat()
                    val clipFile = row.optStringOrNull("clip_file")
                    grindEvents += GrindEvent(
                        id = row.optString("id", "${sessionId}_$i"),
                        offsetS = startS,
                        timestamp = startedAt.plusSeconds(startS.toLong()),
                        durationS = row.optDouble("duration_s", 0.0).toFloat(),
                        confidence = row.optString("confidence_level").toConfidence(),
                        clipPath = clipFile?.let { File(clipRoot, it).absolutePath },
                    )
                }
            }

            ParsedNight(
                sessionId = json.optString("session_id", sessionId),
                startedAt = startedAt,
                endedAt = endedAt,
                grindEvents = grindEvents,
                totalDurationS = json.optDouble("total_duration_s", 0.0),
                snoreRatio = summary?.optDouble("snore_ratio")?.toFloat(),
            )
        }.getOrNull()
    }

    private data class ParsedNight(
        val sessionId: String,
        val startedAt: Instant,
        val endedAt: Instant,
        val grindEvents: List<GrindEvent>,
        val totalDurationS: Double,
        val snoreRatio: Float?,
    ) {
        fun toNightResult(comparison: Float?): NightResult {
            val durationS = totalDurationS.takeIf { it > 0.0 }
                ?: Duration.between(startedAt, endedAt).seconds.coerceAtLeast(1L).toDouble()
            return NightResult(
                sessionId = sessionId,
                startedAt = startedAt,
                endedAt = endedAt,
                grindEventCount = grindEvents.size,
                eventsPerHour = (grindEvents.size / (durationS / 3600.0)).toFloat(),
                totalGrindDurationS = grindEvents.sumOf { it.durationS.toDouble() }.toFloat(),
                events = grindEvents,
                snoreRatio = snoreRatio,
                comparisonPrevious7d = comparison,
            )
        }
    }
}

private fun String.toConfidence(): Confidence = when (uppercase()) {
    "HIGH" -> Confidence.HIGH
    "LOW" -> Confidence.LOW
    else -> Confidence.MEDIUM
}

private fun String.toInstantOrNull(): Instant? =
    runCatching { OffsetDateTime.parse(this).toInstant() }.getOrNull()
        // session.json is the source of these stamps and normally carries an
        // offset, but an imported/external session may not. Reading it in the
        // phone's own zone is a far better answer than discarding the night.
        ?: runCatching { LocalDateTime.parse(this).atZone(ZoneId.systemDefault()).toInstant() }
            .getOrNull()

private fun JSONObject.optStringOrNull(key: String): String? =
    if (!has(key) || isNull(key)) null else optString(key).takeIf { it.isNotBlank() }
