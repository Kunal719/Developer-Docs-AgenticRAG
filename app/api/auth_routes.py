from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse

from app.auth.google import create_google_flow
# from app.auth.dependencies import get_current_user

from google.oauth2 import id_token
from google.auth.transport import requests

router = APIRouter(prefix="/auth", tags=["auth"])


# @router.get("/me")
# def get_me(user_id: str = Depends(get_current_user)):
#     return {"user_id": user_id}


@router.get("/google/login")
def google_login(request: Request):
    flow = create_google_flow()

    authorization_url, state = flow.authorization_url(
        access_type="online",
        prompt="select_account"
    ) 

    request.session["google_oauth_state"] = state
    request.session["google_code_verifier"] = flow.code_verifier

    return RedirectResponse(authorization_url)

@router.get("/google/callback")
def google_callback(request: Request):
    state = request.session.get("google_oauth_state")
    code_verifier = request.session.get("google_code_verifier") 

    if not state:
        raise HTTPException(
            status_code=400,
            detail="Missing OAuth State"
        )

    flow = create_google_flow()

    flow.fetch_token(
        authorization_response=str(request.url),
        code_verifier = code_verifier
    )

    google_id_token = flow.credentials.id_token

    google_user = id_token.verify_oauth2_token(
        google_id_token,
        requests.Request(),
        flow.client_config["client_id"]
    )

    rag_service = request.app.state.rag_service
    user_id = rag_service.user_store.get_or_create_user(google_user["sub"])

    request.session["user_id"] = user_id

    request.session.pop("google_oauth_state", None)
    request.session.pop("google_code_verifier", None)

    return  RedirectResponse(
        url="http://127.0.0.1:8501",
        status_code=303,
    )