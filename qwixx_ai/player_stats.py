"""Persistent human player profiles and completed-game statistics."""
import json
import hashlib
import hmac
import os
import secrets
import smtplib
from email.message import EmailMessage
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
        self.db.execute("CREATE TABLE IF NOT EXISTS auth_codes (email TEXT PRIMARY KEY, digest TEXT NOT NULL, expires REAL NOT NULL, attempts INTEGER NOT NULL, sent REAL NOT NULL, name TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS auth_sessions (token_hash TEXT PRIMARY KEY, player_id TEXT NOT NULL, expires REAL NOT NULL)")
        columns={r[1] for r in self.db.execute("PRAGMA table_info(players)")}
        if "email" not in columns:self.db.execute("ALTER TABLE players ADD COLUMN email TEXT")
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_players_email ON players(email) WHERE email IS NOT NULL")
        self.db.commit()


    @staticmethod
    def normalize_email(email):
        if not isinstance(email,str):
            raise ValueError("Invalid email")
        email=email.strip().lower()
        if len(email)>254 or "@" not in email or any(c.isspace() for c in email):
            raise ValueError("Invalid email")
        local,domain=email.rsplit("@",1)
        if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
            raise ValueError("Invalid email")
        return email

    def request_code(self,email,name=None):
        email=self.normalize_email(email)
        if name is not None:
            if not isinstance(name,str) or not 1<=len(name.strip())<=40:
                raise ValueError("Name must be 1–40 characters")
            name=name.strip()
        host=os.getenv("QWIXX_SMTP_HOST")
        sender=os.getenv("QWIXX_SMTP_FROM")
        if not host or not sender:
            raise RuntimeError("Email verification is not configured on the server")
        now=time.time()
        with self.lock:
            previous=self.db.execute("SELECT sent FROM auth_codes WHERE email=?",(email,)).fetchone()
            if previous and now-previous[0]<60:
                raise ValueError("Please wait 60 seconds before requesting another code")
            code=f"{secrets.randbelow(1000000):06d}"
            digest=hashlib.sha256((email+":"+code).encode()).hexdigest()
            message=EmailMessage()
            message["From"]=sender;message["To"]=email;message["Subject"]="Qwixx AI verification code"
            message.set_content(f"Your Qwixx AI sign-in code is {code}. It expires in 10 minutes. If you did not request this, ignore this message.")
            port=int(os.getenv("QWIXX_SMTP_PORT","587"))
            if os.getenv("QWIXX_SMTP_SSL","false").lower()=="true":
                smtp=smtplib.SMTP_SSL(host,port,timeout=15)
            else:
                smtp=smtplib.SMTP(host,port,timeout=15)
            with smtp:
                if os.getenv("QWIXX_SMTP_SSL","false").lower()!="true":
                    smtp.starttls()
                user=os.getenv("QWIXX_SMTP_USER")
                if user:smtp.login(user,os.getenv("QWIXX_SMTP_PASSWORD",""))
                smtp.send_message(message)
            self.db.execute("INSERT OR REPLACE INTO auth_codes VALUES (?,?,?,?,?,?)",(email,digest,now+600,0,now,name))
            self.db.commit()
        return {"sent":True}

    def verify_code(self,email,code):
        email=self.normalize_email(email)
        if not isinstance(code,str) or len(code)!=6 or not code.isdigit():
            raise ValueError("Invalid verification code")
        now=time.time()
        with self.lock:
            row=self.db.execute("SELECT digest,expires,attempts,name FROM auth_codes WHERE email=?",(email,)).fetchone()
            if not row or now>row[1] or row[2]>=5:
                raise ValueError("Code expired or too many attempts; request another")
            digest=hashlib.sha256((email+":"+code).encode()).hexdigest()
            if not hmac.compare_digest(digest,row[0]):
                self.db.execute("UPDATE auth_codes SET attempts=attempts+1 WHERE email=?",(email,))
                self.db.commit()
                raise ValueError("Incorrect code")
            player=self.db.execute("SELECT id,name FROM players WHERE email=?",(email,)).fetchone()
            if player is None:
                name=row[3] or email.split("@")[0][:40]
                existing=self.db.execute("SELECT id FROM players WHERE name=? COLLATE NOCASE",(name,)).fetchone()
                if existing:name=(name[:31]+"-"+secrets.token_hex(3))
                pid=uuid.uuid4().hex
                self.db.execute("INSERT INTO players (id,name,created,email) VALUES (?,?,?,?)",(pid,name,now,email))
                player=(pid,name)
            self.db.execute("DELETE FROM auth_codes WHERE email=?",(email,))
            token=secrets.token_urlsafe(32)
            self.db.execute("INSERT INTO auth_sessions VALUES (?,?,?)",(hashlib.sha256(token.encode()).hexdigest(),player[0],now+30*86400))
            self.db.commit()
            return {"player":{"id":player[0],"name":player[1]},"token":token}

    def authenticate(self,token):
        if not token:return None
        digest=hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            row=self.db.execute("SELECT player_id FROM auth_sessions WHERE token_hash=? AND expires>?",(digest,time.time())).fetchone()
            return row[0] if row else None

    def logout(self,token):
        with self.lock:
            self.db.execute("DELETE FROM auth_sessions WHERE token_hash=?",(hashlib.sha256(token.encode()).hexdigest(),))
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
