package com.bruxecho.recorder.domain

/**
 * The only thing UI screens are allowed to know about where detection
 * results come from. `MockDetectionRepository` is the only implementation
 * this sprint (no detection model exists yet); a `RealDetectionRepository`
 * reading actual events.csv output lands in Sprint 6 behind this same
 * interface — no screen should need to change when that happens.
 */
interface DetectionRepository {
    suspend fun getNightResult(sessionId: String): NightResult?
    suspend fun getRecentNights(limit: Int): List<NightSummary>
}
