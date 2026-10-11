"""Persistent human player profiles and completed-game statistics."""
import json
import sqlite3
import threading
import time
import uuid


class PlayerStats:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=20)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS players (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, created REAL NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS results (
            session_id TEXT PRIMARY KEY, player_id TEXT NOT NULL, difficulty TEXT NOT NULL,
            score INTEGER NOT NULL, penalties INTEGER NOT NULL, turns INTEGER NOT NULL,
            scores TEXT NOT NULL, result TEXT NOT NULL, created REAL NOT NULL,
            FOREIGN KEY(player_id) REFERENCES players(id))""")
        self.db.commit()

    def profiles(self):
        with self.lock:
            return [{"id": row[0], "name": row[1]} for row in self.db.execute(
                "SELECT id,name FROM players ORDER BY name COLLATE NOCASE")]

    def create_profile(self, name):
        if not isinstance(name, str):
            raise ValueError("Player name must be text")
        name = name.strip()
        if not 1 <= len(name) <= 40 or any(ord(c) < 32 for c in name):
            raise ValueError("Player name must contain 1–40 printable characters")
        with self.lock:
            try:
                pid = uuid.uuid4().hex
                self.db.execute("INSERT INTO players VALUES (?,?,?)", (pid, name, time.time()))
                self.db.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError("That player name already exists") from exc
            return {"id": pid, "name": name}

    def profile(self, pid):
        with self.lock:
            row = self.db.execute("SELECT id,name FROM players WHERE id=?", (pid,)).fetchone()
            if row is None:
                raise ValueError("Unknown player profile")
            return {"id": row[0], "name": row[1]}

    def record(self, session_id, player_id, difficulty, game):
        if not game.done:
            return False
        self.profile(player_id)
        scores = game.scores()
        highest = max(scores)
        result = "tie" if scores[0] == highest and scores.count(highest) > 1 else ("win" if scores[0] == highest else "loss")
        with self.lock:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, player_id, difficulty, scores[0], game.boards[0].penalties,
                 game.turn + 1, json.dumps(scores), result, time.time()))
            self.db.commit()
            return cursor.rowcount == 1

    def stats(self, pid, difficulty=None, limit=20):
        self.profile(pid)
        if difficulty is not None and difficulty not in ("beginner", "intermediate", "advanced", "champion"):
            raise ValueError("Invalid difficulty")
        limit = max(1, min(int(limit), 100))
        where = "WHERE player_id=?" + (" AND difficulty=?" if difficulty else "")
        args = (pid, difficulty) if difficulty else (pid,)
        with self.lock:
            rows = self.db.execute(
                "SELECT difficulty,score,penalties,turns,scores,result,created FROM results "
                + where + " ORDER BY created ASC, rowid ASC", args).fetchall()
        games = len(rows)
        wins = sum(row[5] == "win" for row in rows)
        ties = sum(row[5] == "tie" for row in rows)
        streak = longest = 0
        for row in rows:
            streak = streak + 1 if row[5] == "win" else 0
            longest = max(longest, streak)
        by_difficulty = {}
        for level in ("beginner", "intermediate", "advanced", "champion"):
            subset = [row for row in rows if row[0] == level]
            if subset:
                by_difficulty[level] = {"games": len(subset), "wins": sum(r[5] == "win" for r in subset),
                                        "average_score": round(sum(r[1] for r in subset) / len(subset), 1)}
        return {
            "player": self.profile(pid), "games": games, "wins": wins,
            "losses": games - wins - ties, "ties": ties,
            "win_rate": round(100 * wins / games, 1) if games else 0,
            "average_score": round(sum(r[1] for r in rows) / games, 1) if games else 0,
            "highest_score": max((r[1] for r in rows), default=None),
            "lowest_score": min((r[1] for r in rows), default=None),
            "average_penalties": round(sum(r[2] for r in rows) / games, 2) if games else 0,
            "average_turns": round(sum(r[3] for r in rows) / games, 1) if games else 0,
            "current_win_streak": streak, "longest_win_streak": longest,
            "by_difficulty": by_difficulty,
            "recent": [{"difficulty": r[0], "score": r[1], "penalties": r[2],
                        "turns": r[3], "scores": json.loads(r[4]), "result": r[5],
                        "created": r[6]} for r in reversed(rows[-limit:])]
        }
