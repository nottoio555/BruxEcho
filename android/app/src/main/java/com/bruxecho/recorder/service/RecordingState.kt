package com.bruxecho.recorder.service

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * In-process shared state between [RecordingService] and the Compose UI.
 * A plain singleton is enough here — the whole app runs in one process, so
 * this avoids the ceremony of a bound-service AIDL/Messenger interface just
 * to push a few numbers to the screen a few times a second.
 */
object RecordingState {
    data class Snapshot(
        val isRecording: Boolean = false,
        val sessionId: String? = null,
        val elapsedTotalS: Double = 0.0,
        val currentRms: Float = 0f,
        val sourceLabel: String = "",
        val sampleRateActual: Int = 0,
        val bytesWrittenTotal: Long = 0,
        val lastError: String? = null,
    )

    private val _state = MutableStateFlow(Snapshot())
    val state = _state.asStateFlow()

    fun update(transform: (Snapshot) -> Snapshot) {
        _state.value = transform(_state.value)
    }

    fun reset() {
        _state.value = Snapshot()
    }
}
