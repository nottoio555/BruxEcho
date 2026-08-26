package com.bruxecho.recorder.audio

import android.os.StatFs
import android.util.Log
import com.bruxecho.recorder.session.InterruptionMeta
import com.bruxecho.recorder.session.SegmentMeta
import com.bruxecho.recorder.session.SessionMeta
import com.bruxecho.recorder.session.SessionStore
import java.io.File
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter
import kotlin.concurrent.thread
import kotlin.math.sqrt

private const val TAG = "RecordingEngine"

private const val SEGMENT_DURATION_S = 3600.0
private const val HEADER_FLUSH_INTERVAL_MS = 30_000L
private const val AUTO_STOP_FREE_BYTES = 500L * 1024 * 1024
private const val PAUSE_POLL_MS = 200L

/**
 * Owns the AudioRecord read loop for one overnight session: reads PCM,
 * writes it to rotating 1-hour WAV segments, keeps session.json current, and
 * reacts to audio-focus pause/resume without losing the session. Runs its
 * own dedicated [Thread] rather than a coroutine — the read loop blocks for
 * up to 8 hours and needs predictable, uninterrupted scheduling.
 */
class RecordingEngine(
    private val sessionDir: File,
    private val requestedSampleRateHz: Int,
) {
    @Volatile private var running = false
    @Volatile private var paused = false
    private var pauseStartedAtMs = 0L

    private lateinit var audioSourceResult: AudioSourceResult
    private lateinit var writer: WavFileWriter
    private lateinit var sessionMeta: SessionMeta
    private var segmentIndex = 0
    private var samplesInCurrentSegment = 0L
    private var thread: Thread? = null

    var onTick: ((elapsedTotalS: Double, rms: Float, bytesTotal: Long) -> Unit)? = null
    var onAutoStopLowStorage: (() -> Unit)? = null
    var onFatalError: ((String) -> Unit)? = null

    val sessionId: String get() = sessionMeta.sessionId
    val sourceLabel: String get() = audioSourceResult.sourceLabel
    val sampleRateActual: Int get() = audioSourceResult.sampleRateActual

    /** Returns false if no audio source could be opened at all. */
    fun start(deviceModel: String, androidVersion: String): Boolean {
        val probeResult = AudioSourceProber.probe(requestedSampleRateHz) ?: return false
        audioSourceResult = probeResult

        val sessionId = sessionDir.name
        sessionMeta = SessionMeta(
            sessionId = sessionId,
            startedAt = nowIso(),
            deviceModel = deviceModel,
            androidVersion = androidVersion,
            audioSourceUsed = probeResult.sourceLabel,
            audioSourceFallbackReason = probeResult.fallbackReason,
            sampleRateRequested = requestedSampleRateHz,
            sampleRateActual = probeResult.sampleRateActual,
            effectsDisabled = probeResult.effectsDisabled.toMutableList(),
            effectsUnavailable = probeResult.effectsUnavailable.toMutableList(),
        )
        openSegment()
        SessionStore.write(sessionMeta, sessionDir)

        audioSourceResult.audioRecord.startRecording()
        running = true
        thread = thread(name = "RecordingEngine") { readLoop() }
        return true
    }

    fun pause(reason: String) {
        if (!running || paused) return
        paused = true
        pauseStartedAtMs = System.currentTimeMillis()
        runCatching { audioSourceResult.audioRecord.stop() }
        sessionMeta.interruptions += InterruptionMeta(at = nowIso(), reason = reason, gapS = 0.0)
        SessionStore.write(sessionMeta, sessionDir)
        Log.w(TAG, "paused: $reason")
    }

    fun resume() {
        if (!running || !paused) return
        val gapS = (System.currentTimeMillis() - pauseStartedAtMs) / 1000.0
        sessionMeta.interruptions.lastOrNull()?.let { last ->
            val idx = sessionMeta.interruptions.lastIndex
            sessionMeta.interruptions[idx] = last.copy(gapS = gapS)
        }
        runCatching { audioSourceResult.audioRecord.startRecording() }
        paused = false
        SessionStore.write(sessionMeta, sessionDir)
        Log.w(TAG, "resumed after ${gapS}s gap")
    }

    fun stop() {
        if (!running) return
        running = false
        thread?.join(5_000)
        thread = null
    }

    private fun readLoop() {
        val chunkShorts = ShortArray(maxOf(audioSourceResult.bufferSizeBytes / 2, 1024))
        val chunkBytes = ByteArray(chunkShorts.size * 2)
        var lastFlushMs = System.currentTimeMillis()
        var totalSamplesWritten = 0L

        try {
            while (running) {
                if (paused) {
                    Thread.sleep(PAUSE_POLL_MS)
                    continue
                }

                val n = audioSourceResult.audioRecord.read(chunkShorts, 0, chunkShorts.size)
                if (n <= 0) {
                    continue
                }

                var sumSquares = 0.0
                for (i in 0 until n) {
                    val sample = chunkShorts[i]
                    sumSquares += (sample.toDouble() / 32768.0) * (sample.toDouble() / 32768.0)
                    val byteIdx = i * 2
                    chunkBytes[byteIdx] = (sample.toInt() and 0xFF).toByte()
                    chunkBytes[byteIdx + 1] = ((sample.toInt() shr 8) and 0xFF).toByte()
                }
                val rms = sqrt(sumSquares / n).toFloat()

                writer.append(chunkBytes, n * 2)
                samplesInCurrentSegment += n
                totalSamplesWritten += n

                if (samplesInCurrentSegment >= audioSourceResult.sampleRateActual.toLong() * SEGMENT_DURATION_S.toLong()) {
                    rotateSegment()
                }

                val now = System.currentTimeMillis()
                if (now - lastFlushMs >= HEADER_FLUSH_INTERVAL_MS) {
                    writer.flushHeader()
                    SessionStore.write(sessionMeta, sessionDir)
                    lastFlushMs = now

                    val freeBytes = StatFs(sessionDir.absolutePath).availableBytes
                    if (freeBytes < AUTO_STOP_FREE_BYTES) {
                        Log.w(TAG, "auto-stop: free space $freeBytes bytes below threshold")
                        onAutoStopLowStorage?.invoke()
                        running = false
                    }
                }

                val elapsedTotalS = sessionMeta.totalDurationS + writer.durationSeconds
                onTick?.invoke(elapsedTotalS, rms, totalSamplesWritten * 2L)
            }
        } catch (e: Exception) {
            Log.e(TAG, "read loop error", e)
            onFatalError?.invoke(e.message ?: "unknown recording error")
        } finally {
            finish()
        }
    }

    private fun openSegment() {
        val file = File(sessionDir, "segment_%03d.wav".format(segmentIndex))
        writer = WavFileWriter(
            file = file,
            sampleRateHz = audioSourceResult.sampleRateActual,
            channels = 1,
            bitDepth = 16,
        )
        samplesInCurrentSegment = 0L
    }

    private fun rotateSegment() {
        val file = File(sessionDir, "segment_%03d.wav".format(segmentIndex))
        writer.close()
        sessionMeta.segments += SegmentMeta(file = file.name, durationS = writer.durationSeconds)
        segmentIndex++
        openSegment()
        SessionStore.write(sessionMeta, sessionDir)
    }

    private fun finish() {
        runCatching {
            val file = File(sessionDir, "segment_%03d.wav".format(segmentIndex))
            writer.close()
            sessionMeta.segments += SegmentMeta(file = file.name, durationS = writer.durationSeconds)
        }
        sessionMeta.endedAt = nowIso()
        SessionStore.write(sessionMeta, sessionDir)
        audioSourceResult.release()
    }

    private fun nowIso(): String = OffsetDateTime.now().format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)
}
