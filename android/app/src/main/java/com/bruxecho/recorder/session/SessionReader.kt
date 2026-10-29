package com.bruxecho.recorder.session

import org.json.JSONObject
import java.io.File

/** Read-only view of a session.json, tolerant of missing/null fields. */
data class ParsedSession(
    val sessionId: String,
    val startedAt: String?,
    val endedAt: String?,
    val deviceModel: String?,
    val androidVersion: String?,
    val audioSourceUsed: String?,
    val audioSourceFallbackReason: String?,
    val sampleRateRequested: Int?,
    val sampleRateActual: Int?,
    val channels: Int?,
    val bitDepth: Int?,
    val effectsDisabled: List<String>,
    val effectsUnavailable: List<String>,
    val segments: List<SegmentMeta>,
    val interruptions: List<InterruptionMeta>,
    val totalDurationS: Double,
    val provenance: String?,
)

/**
 * One row for the sessions list. [parsed] is null when session.json is
 * missing or fails to parse — the app was killed before the very first
 * write, or the file was damaged/tampered with. The UI surfaces that as
 * "บันทึกไม่สมบูรณ์" rather than crashing or silently hiding the night.
 */
data class SessionListEntry(
    val sessionId: String,
    val dir: File,
    val parsed: ParsedSession?,
    val totalWavBytes: Long,
)

object SessionReader {

    fun listSessions(sessionsRoot: File): List<SessionListEntry> {
        val dirs = sessionsRoot.listFiles { f -> f.isDirectory } ?: return emptyList()
        return dirs
            .sortedByDescending { it.name } // session_id is yyyyMMdd_HHmmss — lexicographic sort is chronological
            .map { dir ->
                SessionListEntry(
                    sessionId = dir.name,
                    dir = dir,
                    parsed = readOne(dir),
                    totalWavBytes = dir.listFiles { f -> f.name.endsWith(".wav") }?.sumOf { it.length() } ?: 0L,
                )
            }
    }

    fun readOne(dir: File): ParsedSession? {
        val file = File(dir, "session.json")
        if (!file.exists()) return null
        return try {
            val json = JSONObject(file.readText())
            ParsedSession(
                sessionId = json.optString("session_id", dir.name),
                startedAt = json.optStringOrNull("started_at"),
                endedAt = json.optStringOrNull("ended_at"),
                deviceModel = json.optStringOrNull("device_model"),
                androidVersion = json.optStringOrNull("android_version"),
                audioSourceUsed = json.optStringOrNull("audio_source_used"),
                audioSourceFallbackReason = json.optStringOrNull("audio_source_fallback_reason"),
                sampleRateRequested = json.optIntOrNull("sample_rate_requested"),
                sampleRateActual = json.optIntOrNull("sample_rate_actual"),
                channels = json.optIntOrNull("channels"),
                bitDepth = json.optIntOrNull("bit_depth"),
                effectsDisabled = json.optStringList("effects_disabled"),
                effectsUnavailable = json.optStringList("effects_unavailable"),
                segments = json.optJSONArray("segments")?.let { arr ->
                    (0 until arr.length()).map { i ->
                        val seg = arr.getJSONObject(i)
                        SegmentMeta(file = seg.getString("file"), durationS = seg.getDouble("duration_s"))
                    }
                } ?: emptyList(),
                interruptions = json.optJSONArray("interruptions")?.let { arr ->
                    (0 until arr.length()).map { i ->
                        val gap = arr.getJSONObject(i)
                        InterruptionMeta(
                            at = gap.getString("at"),
                            reason = gap.getString("reason"),
                            gapS = gap.optDouble("gap_s", 0.0),
                        )
                    }
                } ?: emptyList(),
                totalDurationS = json.optDouble("total_duration_s", 0.0),
                provenance = json.optStringOrNull("provenance"),
            )
        } catch (e: Exception) {
            null
        }
    }

    fun deleteSession(dir: File): Boolean = dir.deleteRecursively()
}

private fun JSONObject.optStringOrNull(key: String): String? =
    if (!has(key) || isNull(key)) null else getString(key)

private fun JSONObject.optIntOrNull(key: String): Int? =
    if (!has(key) || isNull(key)) null else getInt(key)

private fun JSONObject.optStringList(key: String): List<String> {
    val arr = optJSONArray(key) ?: return emptyList()
    return (0 until arr.length()).map { arr.getString(it) }
}
