package com.bruxecho.recorder.audio

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min

data class MicCheckResult(
    val sourceLabel: String,
    val sampleRateRequested: Int,
    val sampleRateActual: Int,
    val fallbackReason: String?,
    val bandEnergyPercent: Double,
    val dspLikelyStillFiltering: Boolean,
)

enum class MicCheckPhase { RECORDING, ANALYZING }

/**
 * Records a short test clip and measures how much energy survives in the
 * 8-16 kHz band — the band Sprint 1 found carries the SNR that separates
 * grinding from silence, and the band ordinary phone-recorder apps and
 * Samsung's voice DSP discard. A near-zero result here means UNPROCESSED
 * was granted at the API level but the DSP is still filtering underneath it.
 */
object MicCheckRecorder {
    private const val TEST_DURATION_S = 5
    private const val LOW_HZ = 8_000.0
    private const val HIGH_HZ = 16_000.0
    private const val DSP_WARNING_THRESHOLD_PCT = 1.0

    suspend fun runCheck(
        requestedSampleRateHz: Int = 44_100,
        onPhase: ((MicCheckPhase) -> Unit)? = null,
    ): MicCheckResult? = withContext(Dispatchers.IO) {
        val probeResult = AudioSourceProber.probe(requestedSampleRateHz) ?: return@withContext null
        try {
            val record = probeResult.audioRecord
            record.startRecording()
            withContext(Dispatchers.Main) { onPhase?.invoke(MicCheckPhase.RECORDING) }

            val totalSamples = probeResult.sampleRateActual * TEST_DURATION_S
            val pcm = ShortArray(totalSamples)
            val chunk = ShortArray(max(probeResult.bufferSizeBytes / 2, 1024))
            var readTotal = 0
            while (readTotal < totalSamples) {
                val toRead = min(chunk.size, totalSamples - readTotal)
                val n = record.read(chunk, 0, toRead)
                if (n <= 0) break
                System.arraycopy(chunk, 0, pcm, readTotal, n)
                readTotal += n
            }
            record.stop()
            withContext(Dispatchers.Main) { onPhase?.invoke(MicCheckPhase.ANALYZING) }

            val captured = if (readTotal == pcm.size) pcm else pcm.copyOf(readTotal)
            val bandPct = FftAnalyzer.bandEnergyRatioPercent(
                captured,
                probeResult.sampleRateActual,
                LOW_HZ,
                HIGH_HZ,
            )

            MicCheckResult(
                sourceLabel = probeResult.sourceLabel,
                sampleRateRequested = requestedSampleRateHz,
                sampleRateActual = probeResult.sampleRateActual,
                fallbackReason = probeResult.fallbackReason,
                bandEnergyPercent = bandPct,
                dspLikelyStillFiltering = bandPct < DSP_WARNING_THRESHOLD_PCT,
            )
        } finally {
            probeResult.release()
        }
    }
}
