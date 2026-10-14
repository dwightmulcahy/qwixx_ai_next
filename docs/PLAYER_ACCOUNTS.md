# Email-verified player accounts

Qwixx AI supports passwordless email verification for human player statistics.

## Configure SMTP

Create a local `.env` file alongside `compose.yaml` (do not commit it). Qwixx uses the same shared SMTP settings as Logwatcher:

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.gmail.addr.here@gmail.com
SMTP_PASS=your-gmail-app-password
QWIXX_COOKIE_SECURE=true
```

Player email uses authenticated SMTP with STARTTLS on port 587. For Gmail, `SMTP_PASS` must be a Google App Password rather than your normal account password. `SMTP_USER` is also used as the message From address. The container loads `.env` directly, so there is no separate Qwixx-specific SMTP credential layer to override these values.

Use `QWIXX_COOKIE_SECURE=true` when serving the app over HTTPS. For local HTTP-only development, leave it false. Never expose a public login over plain HTTP.

Restart with `docker compose up -d --build --force-recreate` after changing `.env`. New players enter their name and email, request a six-digit code, and verify it. Existing registered players request a code for the same email without needing to enter a name. Codes expire in 10 minutes; each email may request another after 60 seconds; five failed attempts invalidate the code. Login cookies expire after 30 days. Guest games are still available but do not count toward stats.

The SQLite database remains at `./runs/players.sqlite3` and is not overwritten by Docker rebuilds. Old unverified player profiles and game histories remain in the database, but are **not automatically assigned to an email**: knowing a display name does not prove account ownership. An administrator must manually migrate existing unverified records after confirming ownership.

Email sending requires an SMTP provider; the application does not include an email delivery service.
