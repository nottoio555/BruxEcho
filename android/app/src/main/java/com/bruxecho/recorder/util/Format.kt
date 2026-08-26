package com.bruxecho.recorder.util

import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.chrono.ThaiBuddhistChronology
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.math.abs

private val THAI_LOCALE = Locale.Builder().setLanguage("th").setRegion("TH").build()

private val THAI_DATE_FORMATTER: DateTimeFormatter =
    DateTimeFormatter.ofPattern("d MMMM yyyy", THAI_LOCALE).withChronology(ThaiBuddhistChronology.INSTANCE)

private val TIME_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm", THAI_LOCALE)
private val TIME_SECONDS_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss", THAI_LOCALE)

/** e.g. "24 สิงหาคม 2569" (Buddhist Era). Falls back to the raw string if it doesn't parse. */
fun formatThaiDate(isoDateTime: String?): String {
    if (isoDateTime == null) return "-"
    return try {
        OffsetDateTime.parse(isoDateTime).format(THAI_DATE_FORMATTER)
    } catch (e: Exception) {
        isoDateTime
    }
}

fun formatTimeOnly(isoDateTime: String?): String {
    if (isoDateTime == null) return "-"
    return try {
        OffsetDateTime.parse(isoDateTime).format(TIME_FORMATTER)
    } catch (e: Exception) {
        "-"
    }
}

fun formatThaiDate(instant: Instant): String = instant.atZone(ZoneId.systemDefault()).format(THAI_DATE_FORMATTER)

fun formatTimeOnly(instant: Instant): String = instant.atZone(ZoneId.systemDefault()).format(TIME_FORMATTER)

fun formatTimeWithSeconds(instant: Instant): String = instant.atZone(ZoneId.systemDefault()).format(TIME_SECONDS_FORMATTER)

/** Long-form duration for list rows, e.g. "7 ชม. 32 นาที" / "45 นาที" / "12 วินาที". */
fun formatDurationThai(totalSeconds: Double): String {
    val total = totalSeconds.toLong().coerceAtLeast(0)
    val h = total / 3600
    val m = (total % 3600) / 60
    val s = total % 60
    return when {
        h > 0 -> "$h ชม. $m นาที"
        m > 0 -> "$m นาที"
        else -> "$s วินาที"
    }
}

fun formatHms(totalSeconds: Double): String {
    val total = totalSeconds.toLong().coerceAtLeast(0)
    val h = total / 3600
    val m = (total % 3600) / 60
    val s = total % 60
    return "%02d:%02d:%02d".format(h, m, s)
}

fun formatBytes(bytes: Long): String {
    if (abs(bytes) < 1024) return "$bytes B"
    val units = listOf("KB", "MB", "GB", "TB")
    var value = bytes.toDouble()
    var unitIndex = -1
    while (value >= 1024 && unitIndex < units.lastIndex) {
        value /= 1024
        unitIndex++
    }
    return "%.1f %s".format(value, units[unitIndex])
}
