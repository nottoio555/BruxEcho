package com.bruxecho.recorder.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.sp

// Thai vowel and tone marks stack above and below the consonant line, so a
// line-height tuned for Latin text (~1.2x) clips them under tight leading.
// 1.6x keeps multi-line Thai body text fully legible — this is the whole
// reason this file defines its own Typography instead of using Material's
// default English-tuned one.
private const val THAI_LINE_HEIGHT_MULTIPLIER = 1.6f

private fun thaiStyle(fontSize: TextUnit, weight: FontWeight = FontWeight.Normal): TextStyle = TextStyle(
    fontSize = fontSize,
    fontWeight = weight,
    lineHeight = (fontSize.value * THAI_LINE_HEIGHT_MULTIPLIER).sp,
)

val BruxEchoTypography = Typography(
    titleLarge = thaiStyle(22.sp, FontWeight.Bold),
    titleMedium = thaiStyle(18.sp, FontWeight.Bold),
    titleSmall = thaiStyle(16.sp, FontWeight.Medium),
    bodyLarge = thaiStyle(16.sp),
    bodyMedium = thaiStyle(14.sp),
    bodySmall = thaiStyle(13.sp),
    labelLarge = thaiStyle(14.sp, FontWeight.Medium),
    labelMedium = thaiStyle(12.sp, FontWeight.Medium),
    labelSmall = thaiStyle(11.sp, FontWeight.Medium),
)

/**
 * Tabular (fixed-width) digits so a changing number doesn't visibly shift
 * its neighbors in place — the recording timer and the result-screen stat
 * counters both update while on screen. Monospace guarantees equal digit
 * widths on its own; `tnum` is added for glyph quality on fonts that
 * support it and is silently ignored otherwise.
 */
fun tabularNumberStyle(fontSize: TextUnit, weight: FontWeight = FontWeight.Bold): TextStyle = TextStyle(
    fontFamily = FontFamily.Monospace,
    fontFeatureSettings = "tnum, lnum",
    fontSize = fontSize,
    fontWeight = weight,
)
