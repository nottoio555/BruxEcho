package com.bruxecho.recorder.session

/** One rotated WAV file within a session (~1 hour each, see [com.bruxecho.recorder.audio.RecordingEngine]). */
data class SegmentMeta(
    val file: String,
    val durationS: Double,
)

/** A gap in the recording, e.g. audio focus lost to a phone call. */
data class InterruptionMeta(
    val at: String,
    val reason: String,
    val gapS: Double,
)

/**
 * Mutable research metadata for one overnight recording session, mirrored to
 * session.json after every state change so a crash mid-night doesn't lose
 * provenance for the segments already on disk.
 */
class SessionMeta(
    val sessionId: String,
    val startedAt: String,
    var endedAt: String? = null,
    val deviceModel: String,
    val androidVersion: String,
    var audioSourceUsed: String,
    var audioSourceFallbackReason: String? = null,
    val sampleRateRequested: Int,
    var sampleRateActual: Int,
    val channels: Int = 1,
    val bitDepth: Int = 16,
    val effectsDisabled: MutableList<String> = mutableListOf(),
    val effectsUnavailable: MutableList<String> = mutableListOf(),
    val segments: MutableList<SegmentMeta> = mutableListOf(),
    val interruptions: MutableList<InterruptionMeta> = mutableListOf(),
) {
    val totalDurationS: Double
        get() = segments.sumOf { it.durationS }
}
