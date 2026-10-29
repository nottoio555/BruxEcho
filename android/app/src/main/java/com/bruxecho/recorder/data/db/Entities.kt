package com.bruxecho.recorder.data.db

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.time.Instant
import java.time.LocalDate

enum class EventType { GRIND, SNORE }
enum class Severity { MILD, SEVERE }
enum class EventConfidence { HIGH, MEDIUM, LOW }
enum class LifestyleTag { CAFFEINE, ALCOHOL, EXERCISE, STRESS, LATE_SLEEP }

/**
 * Mirrors the subset of session.json this app actually reads. The audio
 * files themselves stay on disk under StorageInfo.sessionsRootDir — this
 * table stores metadata only, never audio bytes.
 */
@Entity(tableName = "sessions")
data class SessionEntity(
    @PrimaryKey val sessionId: String,
    val startedAt: Instant,
    val endedAt: Instant?,
    val audioSource: String?,
    val sampleRate: Int?,
    val segmentCount: Int,
    val totalDurationS: Double,
    val isComplete: Boolean,
    val provenance: String,
)

/**
 * A detected grind or snore event within a session. No real detection model
 * exists yet (see CLAUDE.md) — until Sprint 6, rows here are deterministically
 * generated mock data (same approach [MockDetectionRepository] used before
 * this table existed), so the same session always shows the same numbers.
 */
@Entity(tableName = "events")
data class EventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val startS: Float,
    val endS: Float,
    val type: EventType,
    val severity: Severity,
    val confidence: EventConfidence,
    val clipPath: String?,
) {
    val durationS: Float get() = endS - startS
}

/** One behavior tag for one calendar date — see the Sprint 4.5 "แท็บพฤติกรรม" feature. */
@Entity(tableName = "lifestyle_tags")
data class LifestyleTagEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val date: LocalDate,
    val tag: LifestyleTag,
    val memo: String?,
)

/**
 * Precomputed per-date rollup, one row per calendar date.
 *
 * The primary key is the date, so this row covers **every** session that
 * started that day, not one of them. That distinction was a real bug: two
 * sessions on the same date used to write this row twice, the second silently
 * replacing the first, and the calendar then showed one night while the
 * trends screen showed the other. Counts here are therefore day totals, and
 * [sessionId] is only the most recent session of the day — a default target
 * for the "open results" button, never the whole story. Use
 * [SessionDao.getStartedBetween] to list a day's sessions.
 *
 * Kept in sync with `sessions`/`events` by [com.bruxecho.recorder.data.SessionSync]
 * — lets the calendar grid and stats screens query 30-90 days at once without
 * re-aggregating events on every frame.
 */
@Entity(tableName = "daily_summary")
data class DailySummaryEntity(
    @PrimaryKey val date: LocalDate,
    val sessionId: String,
    val sleepDurationS: Double,
    val mildCount: Int,
    val severeCount: Int,
    val snoreLightCount: Int,
    val snoreLoudCount: Int,
    val isComplete: Boolean,
)

/** One row of the calendar's per-day session list. */
data class DaySession(
    val session: SessionEntity,
    val grindCount: Int,
)
