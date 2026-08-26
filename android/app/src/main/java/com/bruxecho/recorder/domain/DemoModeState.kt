package com.bruxecho.recorder.domain

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

private const val PREFS_NAME = "bruxecho_prefs"
private const val KEY_DEMO_MODE = "demo_mode_enabled"

/**
 * Whether the app should show curated sample data instead of real session
 * data — for presentations, so a demo never depends on whether last
 * night's recording actually happened or succeeded (see settings screen).
 * Persisted so it survives restarts; exposed as Compose state so flipping
 * the switch updates every open screen immediately.
 */
object DemoModeState {
    var isEnabled by mutableStateOf(false)
        private set

    fun init(context: Context) {
        isEnabled = prefs(context).getBoolean(KEY_DEMO_MODE, false)
    }

    fun setEnabled(context: Context, enabled: Boolean) {
        isEnabled = enabled
        prefs(context).edit().putBoolean(KEY_DEMO_MODE, enabled).apply()
    }

    private fun prefs(context: Context) = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
}
