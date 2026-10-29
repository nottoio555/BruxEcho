package com.bruxecho.recorder.domain

/**
 * [DetectionRepository] plus the two session-lookup helpers the results
 * screen needs (which night to show, and whether it's even analyzable).
 *
 * Deliberately a *sub*-interface rather than extra methods on
 * [DetectionRepository]: that interface is the contract every screen is
 * written against and stays untouched, while implementations that can answer
 * "which session" questions declare it here.
 */
interface SessionAwareDetectionRepository : DetectionRepository {

    /** Whether a session exists but can't produce a result (crashed / never finished). */
    suspend fun isSessionIncomplete(sessionId: String): Boolean

    /**
     * The most recent session, complete or not — deliberately does not skip
     * past an incomplete one to find an older working result.
     */
    suspend fun latestSessionId(): String?
}
