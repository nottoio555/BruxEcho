package com.bruxecho.recorder.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(
    entities = [
        SessionEntity::class,
        EventEntity::class,
        LifestyleTagEntity::class,
        DailySummaryEntity::class,
    ],
    version = 1,
    exportSchema = false,
)
@TypeConverters(Converters::class)
abstract class BruxEchoDatabase : RoomDatabase() {
    abstract fun sessionDao(): SessionDao
    abstract fun eventDao(): EventDao
    abstract fun lifestyleTagDao(): LifestyleTagDao
    abstract fun dailySummaryDao(): DailySummaryDao

    companion object {
        @Volatile private var instance: BruxEchoDatabase? = null

        fun getInstance(context: Context): BruxEchoDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    BruxEchoDatabase::class.java,
                    "bruxecho.db",
                ).build().also { instance = it }
            }
    }
}
