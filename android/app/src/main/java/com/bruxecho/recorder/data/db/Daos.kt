package com.bruxecho.recorder.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Upsert
import java.time.LocalDate
import kotlinx.coroutines.flow.Flow

@Dao
interface SessionDao {
    @Upsert
    suspend fun upsert(session: SessionEntity)

    @Query("SELECT * FROM sessions WHERE sessionId = :sessionId")
    suspend fun getBySessionId(sessionId: String): SessionEntity?

    @Query("SELECT sessionId FROM sessions")
    suspend fun getAllIds(): List<String>

    @Query("SELECT sessionId FROM sessions WHERE isComplete = 1")
    suspend fun getCompleteIds(): List<String>

    @Query("SELECT * FROM sessions ORDER BY startedAt DESC LIMIT :limit")
    suspend fun getRecent(limit: Int): List<SessionEntity>

    @Query("SELECT * FROM sessions ORDER BY startedAt DESC LIMIT 1")
    suspend fun getLatest(): SessionEntity?

    /**
     * Sessions that started within a half-open instant range.
     *
     * Takes epoch millis rather than a LocalDate because `sessions` stores an
     * instant: a "day" is only a day in some timezone, and the caller is the
     * one that knows which. Half-open so midnight belongs to exactly one day.
     */
    @Query("SELECT * FROM sessions WHERE startedAt >= :startMs AND startedAt < :endMs ORDER BY startedAt")
    suspend fun getStartedBetween(startMs: Long, endMs: Long): List<SessionEntity>

    @Query("SELECT COUNT(*) FROM sessions")
    fun observeCount(): Flow<Int>

    @Query("DELETE FROM sessions WHERE sessionId = :sessionId")
    suspend fun deleteBySessionId(sessionId: String)
}

@Dao
interface EventDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(events: List<EventEntity>)

    @Query("SELECT * FROM events WHERE sessionId = :sessionId ORDER BY startS")
    suspend fun getForSession(sessionId: String): List<EventEntity>

    @Query("DELETE FROM events WHERE sessionId = :sessionId")
    suspend fun deleteForSession(sessionId: String)

    @Query("SELECT COUNT(*) FROM events WHERE sessionId = :sessionId AND type = :type")
    suspend fun countForSession(sessionId: String, type: EventType): Int
}

@Dao
interface LifestyleTagDao {
    @Insert
    suspend fun insertAll(tags: List<LifestyleTagEntity>)

    @Query("SELECT * FROM lifestyle_tags WHERE date = :date")
    suspend fun getForDate(date: LocalDate): List<LifestyleTagEntity>

    @Query("SELECT * FROM lifestyle_tags WHERE date >= :since ORDER BY date DESC")
    fun observeSince(since: LocalDate): Flow<List<LifestyleTagEntity>>

    @Query("SELECT * FROM lifestyle_tags WHERE date >= :since ORDER BY date DESC")
    suspend fun getSince(since: LocalDate): List<LifestyleTagEntity>

    @Query("DELETE FROM lifestyle_tags WHERE date = :date")
    suspend fun deleteForDate(date: LocalDate)
}

@Dao
interface DailySummaryDao {
    @Upsert
    suspend fun upsert(summary: DailySummaryEntity)

    @Query("SELECT * FROM daily_summary WHERE date = :date")
    suspend fun getForDate(date: LocalDate): DailySummaryEntity?

    @Query("SELECT * FROM daily_summary WHERE date BETWEEN :start AND :end ORDER BY date")
    fun observeRange(start: LocalDate, end: LocalDate): Flow<List<DailySummaryEntity>>

    @Query("SELECT * FROM daily_summary WHERE date BETWEEN :start AND :end ORDER BY date")
    suspend fun getRange(start: LocalDate, end: LocalDate): List<DailySummaryEntity>

    @Query("DELETE FROM daily_summary WHERE sessionId = :sessionId")
    suspend fun deleteBySessionId(sessionId: String)
}
