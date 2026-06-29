from pfund.entities.accounts import APIKeyAccount


class BybitAccount(APIKeyAccount):
    # NOTE: only supports unified accounts
    @property
    def type(self) -> str:
        return "UNIFIED"
