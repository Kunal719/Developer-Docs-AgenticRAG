from fastapi import HTTPException, Request

def get_current_user(request: Request) -> str:
    user_id = request.session.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return user_id