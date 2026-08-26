package com.bruxecho.recorder.session

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Serializes [SessionMeta] to session.json. Writes to a temp file and renames
 * over the target so a crash mid-write never leaves a truncated/corrupt file
 * behind for the still-open WAV segments to be associated with.
 */
object SessionStore {

    /**
     * org.json's [JSONObject.put] silently *removes* the key when the value
     * is a Kotlin `null` (Android's implementation, unlike the reference
     * org.json), instead of writing JSON `null`. That would make
     * `ended_at`/`audio_source_fallback_reason` disappear entirely from an
     * in-progress session.json instead of reading as `null` — breaking
     * anything that expects the key to always be present.
     */
    private fun JSONObject.putNullable(key: String, value: String?) {
        put(key, value ?: JSONObject.NULL)
    }

    fun write(meta: SessionMeta, sessionDir: File) {
        val json = JSONObject().apply {
            put("session_id", meta.sessionId)
            put("started_at", meta.startedAt)
            putNullable("ended_at", meta.endedAt)
            put("device_model", meta.deviceModel)
            put("android_version", meta.androidVersion)
            put("audio_source_used", meta.audioSourceUsed)
            putNullable("audio_source_fallback_reason", meta.audioSourceFallbackReason)
            put("sample_rate_requested", meta.sampleRateRequested)
            put("sample_rate_actual", meta.sampleRateActual)
            put("channels", meta.channels)
            put("bit_depth", meta.bitDepth)
            put("effects_disabled", JSONArray(meta.effectsDisabled))
            put("effects_unavailable", JSONArray(meta.effectsUnavailable))
            put("segments", JSONArray().apply {
                meta.segments.forEach { seg ->
                    put(JSONObject().apply {
                        put("file", seg.file)
                        put("duration_s", seg.durationS)
                    })
                }
            })
            put("interruptions", JSONArray().apply {
                meta.interruptions.forEach { gap ->
                    put(JSONObject().apply {
                        put("at", gap.at)
                        put("reason", gap.reason)
                        put("gap_s", gap.gapS)
                    })
                }
            })
            put("total_duration_s", meta.totalDurationS)
        }

        val target = File(sessionDir, "session.json")
        val tmp = File(sessionDir, "session.json.tmp")
        tmp.writeText(json.toString(2))
        if (!tmp.renameTo(target)) {
            // Cross-filesystem or locked-file fallback: copy then delete.
            target.writeText(tmp.readText())
            tmp.delete()
        }
    }
}
