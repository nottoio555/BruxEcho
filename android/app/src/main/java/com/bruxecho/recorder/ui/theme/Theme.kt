package com.bruxecho.recorder.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val BruxEchoDarkScheme = darkColorScheme(
    primary = Accent,
    onPrimary = OnAccent,
    secondary = Warn,
    background = BgDeep,
    onBackground = TextPrimary,
    surface = Surface1,
    onSurface = TextPrimary,
    surfaceVariant = Surface2,
    onSurfaceVariant = TextSecondary,
    error = ErrorRed,
)

/**
 * Always dark, regardless of system theme — this app is used right before
 * sleep, where a bright screen defeats the purpose. There is intentionally
 * no light variant or theme toggle.
 *
 * The accent is warm amber/orange, not blue. Blue-heavy light suppresses
 * melatonin release more than warm light at the same brightness — the
 * eye's circadian photoreceptors are most sensitive around that
 * wavelength — so a dark theme that still leans on blue for its highlights
 * undermines the point of a pre-sleep app.
 */
@Composable
fun BruxEchoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = BruxEchoDarkScheme,
        typography = BruxEchoTypography,
        content = content,
    )
}
