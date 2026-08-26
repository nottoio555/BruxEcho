package com.bruxecho.recorder.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AudioEffect
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor

/**
 * Result of successfully opening an [AudioRecord]. [effectInstances] must be
 * kept referenced (and [release]d) for as long as recording continues —
 * letting them get garbage-collected can re-enable the platform effect.
 */
class AudioSourceResult(
    val audioRecord: AudioRecord,
    val sourceConstant: Int,
    val sourceLabel: String,
    val sampleRateActual: Int,
    val fallbackReason: String?,
    val bufferSizeBytes: Int,
    val effectsDisabled: List<String>,
    val effectsUnavailable: List<String>,
    private val effectInstances: List<AudioEffect>,
) {
    fun release() {
        effectInstances.forEach { runCatching { it.release() } }
        runCatching { audioRecord.release() }
    }
}

/**
 * Picks the least-processed audio source the device will actually
 * initialize with, per BruxEcho's Sprint 1 finding that Samsung/OEM DSP
 * pipelines strip content above 8 kHz — exactly the band that separates
 * grinding from silence. UNPROCESSED is tried first because it bypasses
 * that pipeline entirely; CAMCORDER is the next-least-processed source on
 * devices that reject UNPROCESSED; MIC (voice-call tuned, heaviest
 * processing) is the last resort so the app still records *something*.
 */
object AudioSourceProber {

    private val SOURCE_CHAIN: List<Pair<Int, String>> = buildList {
        add(MediaRecorder.AudioSource.UNPROCESSED to "UNPROCESSED")
        add(MediaRecorder.AudioSource.CAMCORDER to "CAMCORDER")
        add(MediaRecorder.AudioSource.MIC to "MIC")
    }

    /**
     * Tries [requestedSampleRateHz] across the whole source chain first; only
     * if every source refuses that rate does it retry the chain at 48 kHz,
     * which is the native rate some Samsung devices force internally even
     * when 44.1 kHz is requested (see README "Samsung battery/DSP notes").
     */
    @SuppressLint("MissingPermission")
    fun probe(requestedSampleRateHz: Int): AudioSourceResult? {
        val ratesToTry = listOf(requestedSampleRateHz, 48_000).distinct()

        for (rate in ratesToTry) {
            for ((sourceConstant, label) in SOURCE_CHAIN) {
                val (record, bufferSize) = tryCreate(sourceConstant, rate) ?: continue

                val fallbackReason = buildFallbackReason(
                    sourceUsed = label,
                    rateUsed = rate,
                    requestedRate = requestedSampleRateHz,
                )
                val (disabled, unavailable, instances) = disableEffects(record.audioSessionId)

                return AudioSourceResult(
                    audioRecord = record,
                    sourceConstant = sourceConstant,
                    sourceLabel = label,
                    sampleRateActual = rate,
                    fallbackReason = fallbackReason,
                    bufferSizeBytes = bufferSize,
                    effectsDisabled = disabled,
                    effectsUnavailable = unavailable,
                    effectInstances = instances,
                )
            }
        }
        return null
    }

    private fun buildFallbackReason(sourceUsed: String, rateUsed: Int, requestedRate: Int): String? {
        val parts = mutableListOf<String>()
        if (sourceUsed != "UNPROCESSED") {
            parts += "audio source UNPROCESSED ไม่พร้อมใช้งาน ใช้ $sourceUsed แทน"
        }
        if (rateUsed != requestedRate) {
            parts += "sample rate $requestedRate Hz ไม่รองรับ ใช้ $rateUsed Hz แทน"
        }
        return if (parts.isEmpty()) null else parts.joinToString("; ")
    }

    @SuppressLint("MissingPermission")
    private fun tryCreate(source: Int, sampleRateHz: Int): Pair<AudioRecord, Int>? {
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuf == AudioRecord.ERROR || minBuf == AudioRecord.ERROR_BAD_VALUE) return null
        val bufferSize = minBuf * 4

        return try {
            val record = AudioRecord(
                source,
                sampleRateHz,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize,
            )
            if (record.state == AudioRecord.STATE_INITIALIZED) {
                record to bufferSize
            } else {
                record.release()
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private data class EffectsResult(
        val disabled: List<String>,
        val unavailable: List<String>,
        val instances: List<AudioEffect>,
    )

    /**
     * Turns off the three platform effects most likely to alter the signal
     * before it reaches our buffer: echo cancellation, noise suppression,
     * and automatic gain control. All three are designed for voice calls,
     * not for preserving faint high-frequency grinding sounds.
     */
    private fun disableEffects(audioSessionId: Int): EffectsResult {
        val disabled = mutableListOf<String>()
        val unavailable = mutableListOf<String>()
        val instances = mutableListOf<AudioEffect>()

        if (AcousticEchoCanceler.isAvailable()) {
            val aec = AcousticEchoCanceler.create(audioSessionId)
            if (aec != null) {
                aec.enabled = false
                instances += aec
                disabled += "AcousticEchoCanceler"
            } else {
                unavailable += "AcousticEchoCanceler"
            }
        } else {
            unavailable += "AcousticEchoCanceler"
        }

        if (NoiseSuppressor.isAvailable()) {
            val ns = NoiseSuppressor.create(audioSessionId)
            if (ns != null) {
                ns.enabled = false
                instances += ns
                disabled += "NoiseSuppressor"
            } else {
                unavailable += "NoiseSuppressor"
            }
        } else {
            unavailable += "NoiseSuppressor"
        }

        if (AutomaticGainControl.isAvailable()) {
            val agc = AutomaticGainControl.create(audioSessionId)
            if (agc != null) {
                agc.enabled = false
                instances += agc
                disabled += "AutomaticGainControl"
            } else {
                unavailable += "AutomaticGainControl"
            }
        } else {
            unavailable += "AutomaticGainControl"
        }

        return EffectsResult(disabled, unavailable, instances)
    }
}
