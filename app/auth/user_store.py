import uuid
from psycopg_pool import ConnectionPool

class UserStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def get_or_create_user(self, google_subject: str) -> str:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT user_id from users where google_subject = %s
                """,
                (google_subject,)
            )

            row = row.fetchone()

            if row:
                user_id = row[0]

                conn.execute(
                    """
                    UPDATE users
                    SET last_login_at = NOW()
                    WHERE user_id = %s
                    """,
                    (user_id,)
                )

                conn.commit()

                return user_id

            user_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO users (user_id, google_subject)
                VALUES (%s, %s)
                """,
                (user_id, google_subject)
            )

            conn.commit()

            return user_id