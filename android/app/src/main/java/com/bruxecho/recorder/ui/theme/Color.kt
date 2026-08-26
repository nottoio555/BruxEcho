package com.bruxecho.recorder.ui.theme

import androidx.compose.ui.graphics.Color

// Warm, low-color-temperature dark palette. The accent is amber/orange,
// deliberately not blue — see BruxEchoTheme's doc comment in Theme.kt for
// why that choice matters for an app used right before sleep.
val BgDeep = Color(0xFF120F0C)
val Surface1 = Color(0xFF1C1712)
val Surface2 = Color(0xFF261F17)

val Accent = Color(0xFFF0923C) // primary — warm amber/orange
val AccentDim = Color(0xFF5A3D22) // low-emphasis amber fill (e.g. selected chip background)
val OnAccent = Color(0xFF2A1503) // text/icon color on top of Accent-filled surfaces

val Warn = Color(0xFFF2C94C) // gold — kept visually distinct from Accent so warnings don't blend into the brand color
val ErrorRed = Color(0xFFE5574A)
val Positive = Color(0xFF7FC97F) // "improved vs. last week" — muted green, not accent-colored, to avoid implying it's a brand highlight
val Negative = ErrorRed // "worsened vs. last week"

val TextPrimary = Color(0xFFF3ECE3)
val TextSecondary = Color(0xFFA79A89)
