package com.bruxecho.recorder.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.sensitivityDataStore by preferencesDataStore(name = "sensitivity_settings")

enum class SensitivityLevel(val k: Float, val bandRatio: Float) {
    LOW(k = 8.0f, bandRatio = 0.5f),
    MEDIUM(k = 6.0f, bandRatio = 0.35f), // matches DEFAULT_K in src/detect/energy_gate.py
    HIGH(k = 4.0f, bandRatio = 0.2f),
    CUSTOM(k = 6.0f, bandRatio = 0.35f), // placeholder — actual custom values come from SensitivityConfig.k/bandRatio
}

data class SensitivityConfig(val level: SensitivityLevel, val k: Float, val bandRatio: Float)

/**
 * Local-only sensitivity preference. **Not yet consumed by any on-device
 * detection** — there is no real-time/on-device detector this sprint (see
 * CLAUDE.md); this stores the choice so Sprint 6's detector can read it
 * without another settings-UI sprint. `k` and `bandRatio` mirror the
 * parameters `src/detect/energy_gate.py`'s `EnergyGate` already takes.
 */
object SensitivitySettings {
    private val KEY_LEVEL = stringPreferencesKey("level")
    private val KEY_CUSTOM_K = floatPreferencesKey("custom_k")
    private val KEY_CUSTOM_BAND_RATIO = floatPreferencesKey("custom_band_ratio")

    fun observe(context: Context): Flow<SensitivityConfig> =
        context.sensitivityDataStore.data.map { prefs -> prefs.toConfig() }

    suspend fun setLevel(context: Context, level: SensitivityLevel) {
        context.sensitivityDataStore.edit { it[KEY_LEVEL] = level.name }
    }

    suspend fun setCustom(context: Context, k: Float, bandRatio: Float) {
        context.sensitivityDataStore.edit { prefs ->
            prefs[KEY_LEVEL] = SensitivityLevel.CUSTOM.name
            prefs[KEY_CUSTOM_K] = k
            prefs[KEY_CUSTOM_BAND_RATIO] = bandRatio
        }
    }

    private fun Preferences.toConfig(): SensitivityConfig {
        val level = this[KEY_LEVEL]?.let { runCatching { SensitivityLevel.valueOf(it) }.getOrNull() } ?: SensitivityLevel.MEDIUM
        return if (level == SensitivityLevel.CUSTOM) {
            SensitivityConfig(
                level = level,
                k = this[KEY_CUSTOM_K] ?: SensitivityLevel.CUSTOM.k,
                bandRatio = this[KEY_CUSTOM_BAND_RATIO] ?: SensitivityLevel.CUSTOM.bandRatio,
            )
        } else {
            SensitivityConfig(level = level, k = level.k, bandRatio = level.bandRatio)
        }
    }
}
