package com.bruxecho.recorder.util

import android.content.Context
import android.os.StatFs
import java.io.File

const val LOW_SPACE_WARNING_BYTES = 5L * 1024 * 1024 * 1024 // 5 GB

object StorageInfo {

    fun sessionsRootDir(context: Context): File {
        val base = context.getExternalFilesDir(null) ?: context.filesDir
        return File(base, "sessions")
    }

    fun freeBytes(context: Context): Long {
        val dir = sessionsRootDir(context).apply { mkdirs() }
        return StatFs(dir.absolutePath).availableBytes
    }

    /** Sessions that have a session.json with at least one finalized segment. */
    fun completedNightsCount(context: Context): Int {
        val root = sessionsRootDir(context)
        val dirs = root.listFiles { f -> f.isDirectory } ?: return 0
        return dirs.count { File(it, "session.json").exists() }
    }
}
