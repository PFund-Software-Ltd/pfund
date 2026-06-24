import hashlib
import hmac

from pfund.typing import AccountT
from pfund.venues._apis.signers.base import BaseSigner


class HmacSigner(BaseSigner[AccountT]):
    """Shared HMAC primitive for venues that sign with an API key/secret pair.

    Stays generic over the account type; concrete venue signers bind it (e.g.
    ``BybitSigner(HmacSigner[APIKeyAccount])``).
    """

    @staticmethod
    def _hmac_sha256(secret: str, message: str) -> str:
        return hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
