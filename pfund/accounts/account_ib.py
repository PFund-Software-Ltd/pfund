from pfund.accounts.account_base import BaseAccount


class IBAccount(BaseAccount):
    def __init__(self, env: str, host: str='', port: int=None, client_id: int=None, acc: str='', **kwargs):
        super().__init__(env, 'IB', acc=acc, host=host, port=port, client_id=client_id, **kwargs)
        if self.env in ['PAPER', 'LIVE']:
            assert host, f'`host` must be provided for {self.env} trading environment'
            assert port, f'`port` must be provided for {self.env} trading environment'
            assert client_id, f'`client_id` must be provided for {self.env} trading environment'
            