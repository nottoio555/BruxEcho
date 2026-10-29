package com.bruxecho.recorder.ui

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bedtime
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.DirectionsRun
import androidx.compose.material.icons.filled.LocalCafe
import androidx.compose.material.icons.filled.SportsBar
import androidx.compose.ui.graphics.vector.ImageVector
import com.bruxecho.recorder.R
import com.bruxecho.recorder.data.db.LifestyleTag

/**
 * Drawn icon for each tag, used on the full-page tag picker.
 *
 * Vector icons rather than emoji: emoji render in whatever font the device
 * ships, so their size, colour and style are outside the app's control and
 * they sit oddly against a dark theme. These take the theme's colour and
 * scale to any size, which is what a page-sized picker needs.
 */
fun LifestyleTag.icon(): ImageVector = when (this) {
    LifestyleTag.CAFFEINE -> Icons.Filled.LocalCafe
    LifestyleTag.ALCOHOL -> Icons.Filled.SportsBar
    LifestyleTag.EXERCISE -> Icons.Filled.DirectionsRun
    LifestyleTag.STRESS -> Icons.Filled.Bolt
    LifestyleTag.LATE_SLEEP -> Icons.Filled.Bedtime
}

/** Emoji form, still used by the compact text lists on the calendar and stats screens. */
fun LifestyleTag.emoji(): String = when (this) {
    LifestyleTag.CAFFEINE -> "☕"
    LifestyleTag.ALCOHOL -> "🍺"
    LifestyleTag.EXERCISE -> "🏃"
    LifestyleTag.STRESS -> "😤"
    LifestyleTag.LATE_SLEEP -> "🌙"
}

fun LifestyleTag.labelRes(): Int = when (this) {
    LifestyleTag.CAFFEINE -> R.string.lifestyle_tag_caffeine
    LifestyleTag.ALCOHOL -> R.string.lifestyle_tag_alcohol
    LifestyleTag.EXERCISE -> R.string.lifestyle_tag_exercise
    LifestyleTag.STRESS -> R.string.lifestyle_tag_stress
    LifestyleTag.LATE_SLEEP -> R.string.lifestyle_tag_late_sleep
}
