package com.bruxecho.recorder.domain

import java.time.Instant
import java.time.LocalDate

enum class Confidence { HIGH, MEDIUM, LOW }

data class GrindEvent(
    val id: String,
    val offsetS: Float, // seconds since night start — used to place this event on the timeline
    val timestamp: Instant,
    val durationS: Float,
    val confidence: Confidence,
    // "asset:///demo_clips/xxx.wav" for bundled demo clips (see DemoModeState),
    // or a filesystem path once a real clip-cutting pipeline exists (Sprint 6).
    val clipPath: String?,
)

data class NightResult(
    val sessionId: String,
    val startedAt: Instant,
    val endedAt: Instant,
    val grindEventCount: Int,
    val eventsPerHour: Float,
    val totalGrindDurationS: Float,
    val events: List<GrindEvent>,
    val snoreRatio: Float?, // 0-1, null when not estimated
    val comparisonPrevious7d: Float?, // fractional change vs. the trailing 7-day average, e.g. -0.18 = 18% fewer events
)

data class NightSummary(
    val sessionId: String,
    val date: LocalDate,
    val grindEventCount: Int,
)
