package com.bruxecho.recorder.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min

/**
 * Minimal dependency-free spectral analyzer for the mic-check screen. Splits
 * the recording into non-overlapping windows, FFTs each, and averages the
 * power spectrum (Welch-style) — enough to answer one question reliably:
 * "does energy above 8 kHz survive, or has the platform DSP already killed
 * it?" Not intended for anything more precise than that.
 */
object FftAnalyzer {

    private const val FRAME_SIZE = 4096 // power of two

    /** Share of total spectral power (0-100) that falls within [lowHz, highHz]. */
    fun bandEnergyRatioPercent(pcm16: ShortArray, sampleRateHz: Int, lowHz: Double, highHz: Double): Double {
        if (pcm16.size < FRAME_SIZE) return 0.0

        val window = hannWindow(FRAME_SIZE)
        val avgPower = DoubleArray(FRAME_SIZE / 2)
        var frameCount = 0

        var offset = 0
        while (offset + FRAME_SIZE <= pcm16.size) {
            val re = DoubleArray(FRAME_SIZE)
            val im = DoubleArray(FRAME_SIZE)
            for (i in 0 until FRAME_SIZE) {
                re[i] = (pcm16[offset + i] / 32768.0) * window[i]
            }
            fft(re, im)
            for (k in avgPower.indices) {
                avgPower[k] += re[k] * re[k] + im[k] * im[k]
            }
            frameCount++
            offset += FRAME_SIZE
        }
        if (frameCount == 0) return 0.0
        for (k in avgPower.indices) avgPower[k] /= frameCount

        val binHz = sampleRateHz.toDouble() / FRAME_SIZE
        var bandPower = 0.0
        var totalPower = 0.0
        for (k in avgPower.indices) {
            val freq = k * binHz
            totalPower += avgPower[k]
            if (freq in lowHz..highHz) bandPower += avgPower[k]
        }
        if (totalPower <= 0.0) return 0.0
        return min(100.0, (bandPower / totalPower) * 100.0)
    }

    private fun hannWindow(n: Int): DoubleArray = DoubleArray(n) { i ->
        0.5 - 0.5 * cos(2.0 * PI * i / (n - 1))
    }

    /** In-place iterative radix-2 Cooley-Tukey FFT. [re]/[im] length must be a power of two. */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tmpRe = re[i]; re[i] = re[j]; re[j] = tmpRe
                val tmpIm = im[i]; im[i] = im[j]; im[j] = tmpIm
            }
        }

        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRe = cos(ang)
            val wIm = kotlin.math.sin(ang)
            var i = 0
            while (i < n) {
                var curWRe = 1.0
                var curWIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curWRe - im[i + k + len / 2] * curWIm
                    val vIm = re[i + k + len / 2] * curWIm + im[i + k + len / 2] * curWRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nextWRe = curWRe * wRe - curWIm * wIm
                    val nextWIm = curWRe * wIm + curWIm * wRe
                    curWRe = nextWRe
                    curWIm = nextWIm
                }
                i += len
            }
            len = len shl 1
        }
    }
}
