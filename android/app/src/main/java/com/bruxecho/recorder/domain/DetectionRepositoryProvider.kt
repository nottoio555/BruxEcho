package com.bruxecho.recorder.domain

import android.content.Context

/**
 * Chooses where a screen's detection results come from, per call.
 *
 * Three states coexist on a real device, so this routes each request rather
 * than picking one implementation once:
 *
 * 1. **Demo mode on** → [MockDetectionRepository]'s curated dataset, always.
 *    A demo must never depend on whether a night happens to be analyzed.
 * 2. **Session has `events.json`** → [RealDetectionRepository], the actual
 *    model output pushed by `tools/process_session.py`.
 * 3. **Neither** → [MockDetectionRepository]'s placeholder numbers, which is
 *    what every night looked like before Sprint 7.
 *
 * Screens keep talking to [SessionAwareDetectionRepository] and never learn
 * which of these answered — that separation is the whole point of the
 * interface, and it's why a night can be analyzed after the fact and start
 * showing real numbers with no UI change.
 */
object DetectionRepositoryProvider {

    fun get(context: Context): SessionAwareDetectionRepository = RoutingDetectionRepository(context)

    private class RoutingDetectionRepository(context: Context) : SessionAwareDetectionRepository {

        private val appContext = context.applicationContext
        private val real by lazy { RealDetectionRepository(appContext) }
        private val mock by lazy { MockDetectionRepository(appContext) }

        private fun useReal(sessionId: String): Boolean =
            !DemoModeState.isEnabled && real.hasResults(sessionId)

        override suspend fun getNightResult(sessionId: String): NightResult? =
            if (useReal(sessionId)) real.getNightResult(sessionId) else mock.getNightResult(sessionId)

        override suspend fun isSessionIncomplete(sessionId: String): Boolean =
            if (useReal(sessionId)) real.isSessionIncomplete(sessionId) else mock.isSessionIncomplete(sessionId)

        override suspend fun latestSessionId(): String? {
            if (DemoModeState.isEnabled) return mock.latestSessionId()
            // Prefer the newest *analyzed* night so opening the results screen
            // lands on real output when any exists; otherwise fall back to the
            // newest recording so an unanalyzed night still shows its state.
            return real.latestSessionId() ?: mock.latestSessionId()
        }

        override suspend fun getRecentNights(limit: Int): List<NightSummary> {
            if (DemoModeState.isEnabled) return mock.getRecentNights(limit)
            val analyzed = real.getRecentNights(limit)
            if (analyzed.size >= limit) return analyzed
            // Top up the trends chart with un-analyzed nights so the history
            // doesn't appear to start on the day analysis was first run.
            val analyzedIds = analyzed.map { it.sessionId }.toSet()
            val remaining = mock.getRecentNights(limit).filter { it.sessionId !in analyzedIds }
            return (analyzed + remaining).sortedByDescending { it.date }.take(limit)
        }
    }
}
