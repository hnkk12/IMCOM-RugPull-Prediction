"""On-chain constants for Uniswap V2 on Ethereum mainnet."""

UNISWAP_V2_FACTORY = "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

ZERO = "0x0000000000000000000000000000000000000000"
DEAD = "0x000000000000000000000000000000000000dead"
BURN_ADDRESSES = {ZERO, DEAD}

# Uniswap V2 locks MINIMUM_LIQUIDITY LP wei at address(0) on the first mint.
MINIMUM_LIQUIDITY = 1000.0

# topic0 (keccak of event signature)
TOPIC_PAIR_CREATED = "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"
TOPIC_SYNC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"
TOPIC_MINT = "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f"
TOPIC_BURN = "0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496"
TOPIC_SWAP = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
TOPIC_OWNERSHIP = "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0"

# Well-known LP lockers (lower-case). Verify each address on Etherscan before
# relying on it, and extend the list as you confirm more lockers.
LP_LOCKERS = {
    "0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214",  # Unicrypt (UniswapV2Locker)
    "0xe2fe530c047f2d85298b07d9333c05737f1435fb",  # Team.Finance
}

# Function signatures commonly abused in scam tokens (setup D). Selectors are
# derived with keccak at runtime (src/contract_features.py), so add signatures here,
# never raw hex. flag name -> signatures (any match sets the flag).
RISKY_FUNCTIONS = {
    "fn_mint": ["mint(address,uint256)", "mint(uint256)"],
    "fn_blacklist": ["blacklist(address)", "addToBlacklist(address)", "blacklistAddress(address,bool)",
                     "addBlackList(address)", "setBlacklist(address,bool)"],
    "fn_set_bots": ["setBots(address[])", "delBot(address)", "blockBots(address[])"],
    "fn_set_fee": ["setTaxFee(uint256)", "setTaxFeePercent(uint256)", "setFee(uint256)",
                   "setFees(uint256,uint256)", "updateFees(uint256,uint256)", "setSellFee(uint256)"],
    "fn_set_max_tx": ["setMaxTxAmount(uint256)", "setMaxTxPercent(uint256)", "setMaxWalletSize(uint256)"],
    "fn_pause": ["pause()", "unpause()"],
    "fn_trading_switch": ["openTrading()", "enableTrading()", "setTrading(bool)", "setTradingEnabled(bool)"],
    "fn_exclude_from_fee": ["excludeFromFee(address)", "includeInFee(address)"],
    "fn_cooldown": ["setCooldownEnabled(bool)"],
    "fn_owner": ["owner()"],
    "fn_renounce": ["renounceOwnership()"],
    "fn_transfer_ownership": ["transferOwnership(address)"],
}
