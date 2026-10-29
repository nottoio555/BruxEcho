package com.bruxecho.recorder.data.db

import androidx.room.TypeConverter
import java.time.Instant
import java.time.LocalDate

class Converters {
    @TypeConverter
    fun fromInstant(value: Instant?): Long? = value?.toEpochMilli()

    @TypeConverter
    fun toInstant(value: Long?): Instant? = value?.let(Instant::ofEpochMilli)

    @TypeConverter
    fun fromLocalDate(value: LocalDate?): String? = value?.toString() // ISO-8601 "yyyy-MM-dd" — sorts lexicographically

    @TypeConverter
    fun toLocalDate(value: String?): LocalDate? = value?.let(LocalDate::parse)

    @TypeConverter
    fun fromEventType(value: EventType?): String? = value?.name

    @TypeConverter
    fun toEventType(value: String?): EventType? = value?.let(EventType::valueOf)

    @TypeConverter
    fun fromSeverity(value: Severity?): String? = value?.name

    @TypeConverter
    fun toSeverity(value: String?): Severity? = value?.let(Severity::valueOf)

    @TypeConverter
    fun fromEventConfidence(value: EventConfidence?): String? = value?.name

    @TypeConverter
    fun toEventConfidence(value: String?): EventConfidence? = value?.let(EventConfidence::valueOf)

    @TypeConverter
    fun fromLifestyleTag(value: LifestyleTag?): String? = value?.name

    @TypeConverter
    fun toLifestyleTag(value: String?): LifestyleTag? = value?.let(LifestyleTag::valueOf)
}
