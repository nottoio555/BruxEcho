package com.bruxecho.recorder.audio

import java.io.File
import java.io.RandomAccessFile

private const val HEADER_SIZE = 44

/**
 * Streams raw PCM16 into a WAV file, keeping the RIFF/data sizes in the
 * 44-byte header up to date via periodic [flushHeader] calls so the file is
 * always a valid, playable WAV even while still being written — a segment
 * left open by a crash mid-night must still be readable, not truncated junk.
 */
class WavFileWriter(
    file: File,
    private val sampleRateHz: Int,
    private val channels: Int,
    private val bitDepth: Int,
) {
    private val raf = RandomAccessFile(file, "rw")
    private var dataBytesWritten: Long = 0

    init {
        raf.setLength(0)
        writeHeader(0)
        raf.seek(HEADER_SIZE.toLong())
    }

    val bytesPerSecond: Int
        get() = sampleRateHz * channels * (bitDepth / 8)

    val durationSeconds: Double
        get() = dataBytesWritten.toDouble() / bytesPerSecond

    fun append(buffer: ByteArray, length: Int) {
        raf.write(buffer, 0, length)
        dataBytesWritten += length
    }

    /** Rewrites the header with current sizes, then seeks back to append position. */
    fun flushHeader() {
        val pos = raf.filePointer
        writeHeader(dataBytesWritten)
        raf.seek(pos)
        raf.fd.sync()
    }

    fun close() {
        flushHeader()
        raf.close()
    }

    private fun writeHeader(dataSize: Long) {
        val byteRate = sampleRateHz * channels * (bitDepth / 8)
        val blockAlign = channels * (bitDepth / 8)
        val riffSize = 36 + dataSize

        raf.seek(0)
        raf.writeBytes("RIFF")
        raf.write(le32(riffSize.toInt()))
        raf.writeBytes("WAVE")
        raf.writeBytes("fmt ")
        raf.write(le32(16)) // fmt chunk size
        raf.write(le16(1)) // PCM
        raf.write(le16(channels))
        raf.write(le32(sampleRateHz))
        raf.write(le32(byteRate))
        raf.write(le16(blockAlign))
        raf.write(le16(bitDepth))
        raf.writeBytes("data")
        raf.write(le32(dataSize.toInt()))
    }

    private fun le16(v: Int): ByteArray = byteArrayOf((v and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte())

    private fun le32(v: Int): ByteArray = byteArrayOf(
        (v and 0xFF).toByte(),
        ((v shr 8) and 0xFF).toByte(),
        ((v shr 16) and 0xFF).toByte(),
        ((v shr 24) and 0xFF).toByte(),
    )
}
