"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from .base_bars import BaseBars
    from ..util.volume_classifier import get_tick_rule_buy_volume
    from ..util.misc import crop_data_frame_in_batches
except ImportError:
    try:
        from base_bars import BaseBars
    except ImportError:
        import sys
        import os
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from base_bars import BaseBars
    
    # Fallback implementations
    def get_tick_rule_buy_volume(close, volume):
        return volume * 0.5
    def crop_data_frame_in_batches(df, chunksize):
        chunks = []
        for start in range(0, len(df), chunksize):
            end = min(start + chunksize, len(df))
            chunks.append(df.iloc[start:end].copy())
        return chunks


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of bar to create. Example: "tick_bars", "volume_bars", "dollar_bars"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric, batch_size)
        
        # Threshold for sampling
        self.threshold = threshold
        
        # Current cumulative value being tracked
        self.cum_value = 0
        
        # Tick rule memory
        self.last_tick_direction = 1

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        super()._reset_cache()
        self.cum_value = 0

    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        For loop which compiles the various bars: tick, volume, or dollar.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        
        # Standardize column names
        if len(data.columns) >= 3:
            data.columns = ['date_time', 'price', 'volume'] + list(data.columns[3:])
        else:
            raise ValueError("Data must have at least 3 columns: date_time, price, volume")
            
        list_bars = []
        
        for row in data.values:
            # Extract row data
            date_time = row[0]
            price = float(row[1])
            volume = float(row[2]) if len(row) > 2 else 1.0
            
            # Update high and low
            self._update_high_low(price)
            
            # Set open price for new bar
            if self.open_price is None:
                self.open_price = price
                
            # Set previous price for tick classification
            if self.prev_price is not None:
                tick_direction = self._apply_tick_rule(price)
            else:
                tick_direction = 1
                
            self.prev_price = price
            self.close_price = price
            
            # Update cumulative statistics
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_volume'] += volume
            dollar_value = price * volume
            self.cum_statistics['cum_dollar_value'] += dollar_value
            
            if tick_direction == 1:
                self.cum_statistics['cum_buy_volume'] += volume
                
            # Update the specific cumulative value based on bar type
            if self.metric == 'tick_bars':
                self.cum_value += 1
            elif self.metric == 'volume_bars':
                self.cum_value += volume
            elif self.metric == 'dollar_bars':
                self.cum_value += dollar_value
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
                
            # Check if we should create a bar
            if self.cum_value >= self.threshold:
                # Create bar
                bar = self._create_bar(
                    date_time=date_time,
                    price=self.close_price,
                    high_price=self.high_price,
                    low_price=self.low_price,
                    open_price=self.open_price
                )
                
                list_bars.append(bar)
                
                # Reset for next bar
                self._reset_cache()
                
        return list_bars
        
    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29-30 of Advances in Financial Machine Learning.
        
        The tick rule classifies trades as buyer-initiated (1) or seller-initiated (-1).
        When price doesn't change, we use the last known tick direction (memory).
        
        :param price: (float) Current price
        :return: (int) 1 if uptick, -1 if downtick, maintains direction if no change
        """
        if self.prev_price is None:
            # Initialize with positive tick for first trade
            self.last_tick_direction = 1
            return 1
        elif price > self.prev_price:
            # Uptick: buyer-initiated
            self.last_tick_direction = 1
            return 1
        elif price < self.prev_price:
            # Downtick: seller-initiated
            self.last_tick_direction = -1
            return -1
        else:
            # No price change: use last known direction (López de Prado, page 29)
            return getattr(self, 'last_tick_direction', 1)


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                  threshold: int = 1000, batch_size: int = 20000000, 
                  verbose: bool = True, to_csv: bool = False, 
                  output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates tick bars: aggregates transactions into bars by number of transactions.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Number of ticks to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Tick bars
    """
    
    bars = StandardBars(metric='tick_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                    threshold: int = 1000, batch_size: int = 20000000, 
                    verbose: bool = True, to_csv: bool = False, 
                    output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates volume bars: aggregates transactions into bars by cumulative volume.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Volume to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Volume bars
    """
    
    bars = StandardBars(metric='volume_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                    threshold: int = 100000, batch_size: int = 20000000, 
                    verbose: bool = True, to_csv: bool = False, 
                    output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    """
    Creates dollar bars: aggregates transactions into bars by cumulative dollar volume.
    
    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                            raw tick data in the format[date_time, price, volume]
    :param threshold: (int) Dollar volume to aggregate into one bar
    :param batch_size: (int) Number of rows to read in from the csv, per batch
    :param verbose: (bool) Flag whether to print message on each processed batch or not
    :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
    :param output_path: (str) Path to results file, if to_csv = True
    :return: (pd.DataFrame or None) Dollar bars
    """
    
    bars = StandardBars(metric='dollar_bars', threshold=threshold, batch_size=batch_size)
    return bars.batch_run(file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)

# ===============================================
# AUTO-CALIBRACIÓN Y MODO 1M PROXY 
# Implementación basada en López de Prado
# ===============================================

def auto_calibrate_dollar_threshold(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], 
                                   target_bars_per_day: int = 50, 
                                   batch_size: int = 20000000,
                                   sample_days: int = 5) -> float:
    """
    Auto-calibra threshold para conseguir exactamente 50 barras por día (recomendación López de Prado)
    
    :param file_path_or_df: Path to csv or DataFrame with tick data
    :param target_bars_per_day: Target number of bars per day (default 50)
    :param batch_size: Number of rows to read per batch
    :param sample_days: Number of days to sample for calibration
    :return: Calibrated threshold value
    """
    
    # Leer una muestra de datos para calibración
    if isinstance(file_path_or_df, pd.DataFrame):
        sample_data = file_path_or_df.head(batch_size * sample_days)
    else:
        sample_data = pd.read_csv(file_path_or_df, nrows=batch_size * sample_days)
    
    # Standardize column names
    if len(sample_data.columns) >= 3:
        sample_data.columns = ['date_time', 'price', 'volume'] + list(sample_data.columns[3:])
    
    # Calcular dollar volume total
    sample_data['dollar_volume'] = sample_data['price'] * sample_data['volume']
    daily_dollar_volume = sample_data['dollar_volume'].sum()
    
    # Estimar días de trading en la muestra
    sample_data['date_time'] = pd.to_datetime(sample_data['date_time'])
    trading_days = sample_data['date_time'].dt.date.nunique()
    
    # Calcular threshold objetivo
    avg_daily_dollar_volume = daily_dollar_volume / max(trading_days, 1)
    target_threshold = avg_daily_dollar_volume / target_bars_per_day
    
    return target_threshold


def get_auto_calibrated_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                                   target_bars_per_day: int = 50,
                                   batch_size: int = 20000000,
                                   verbose: bool = True,
                                   to_csv: bool = False,
                                   output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Creates auto-calibrated dollar bars targeting 50 bars per day (López de Prado recommendation)
    
    :param file_path_or_df: Path to csv or DataFrame with tick data
    :param target_bars_per_day: Target number of bars per day (default 50)
    :param batch_size: Number of rows to read per batch
    :param verbose: Print progress messages
    :param to_csv: Save results to CSV
    :param output_path: Output file path if to_csv=True
    :return: Auto-calibrated dollar bars DataFrame
    """
    
    # Auto-calibrar threshold
    threshold = auto_calibrate_dollar_threshold(
        file_path_or_df, target_bars_per_day, batch_size
    )
    
    if verbose:
        print(f"Auto-calibrated threshold: ${threshold:,.0f} per bar (target: {target_bars_per_day} bars/day)")
    
    # Crear barras con threshold calibrado
    return get_dollar_bars(file_path_or_df, threshold, batch_size, verbose, to_csv, output_path)


def resample_to_1min_proxy(tick_data: pd.DataFrame, method: str = 'weighted') -> pd.DataFrame:
    """
    Convierte tick data costoso a 1m data como proxy para microestructura
    (Como propuesto en López de Prado's doctoral thesis)
    
    :param tick_data: DataFrame con columns ['date_time', 'price', 'volume'] 
    :param method: 'ohlcv', 'weighted', o 'last_tick'
    :return: 1m bars que preservan características microestructurales
    """
    
    # Standardize column names
    if len(tick_data.columns) >= 3:
        tick_data.columns = ['date_time', 'price', 'volume'] + list(tick_data.columns[3:])
    
    # Asegurar que date_time es datetime
    tick_data['date_time'] = pd.to_datetime(tick_data['date_time'])
    tick_data = tick_data.set_index('date_time')
    
    if method == 'ohlcv':
        # Método tradicional OHLCV
        minute_bars = tick_data.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }).dropna()
        
        minute_bars.columns = ['open', 'high', 'low', 'close', 'volume']
        
    elif method == 'weighted':
        # Método ponderado por volumen (mejor proxy microestructural)
        def weighted_agg(group):
            if len(group) == 0:
                return pd.Series({
                    'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
                    'volume': 0, 'vwap': np.nan, 'tick_count': 0,
                    'buy_volume': 0, 'sell_volume': 0
                })
            
            # Calcular buy/sell volume usando tick rule
            price_changes = group['price'].diff()
            buy_mask = price_changes > 0
            sell_mask = price_changes < 0
            
            return pd.Series({
                'open': group['price'].iloc[0],
                'high': group['price'].max(),
                'low': group['price'].min(),
                'close': group['price'].iloc[-1],
                'volume': group['volume'].sum(),
                'vwap': (group['price'] * group['volume']).sum() / group['volume'].sum() if group['volume'].sum() > 0 else group['price'].iloc[-1],
                'tick_count': len(group),
                'buy_volume': group.loc[buy_mask, 'volume'].sum(),
                'sell_volume': group.loc[sell_mask, 'volume'].sum()
            })
        
        minute_bars = tick_data.resample('1min').apply(weighted_agg).dropna()
        
    elif method == 'last_tick':
        # Método simple: último tick por minuto
        minute_bars = tick_data.resample('1min').last().dropna()
        minute_bars['tick_count'] = tick_data.resample('1min').size()
    
    else:
        raise ValueError("Method must be 'ohlcv', 'weighted', or 'last_tick'")
    
    minute_bars.reset_index(inplace=True)
    return minute_bars


def create_microstructural_features_1m(minute_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características microestructurales usando 1m data como proxy
    Basado en López de Prado's research
    
    :param minute_data: DataFrame con 1m bars
    :return: DataFrame con features microestructurales
    """
    
    features = minute_data.copy()
    
    # 1. Volume-Price Features (capturan order flow)
    if 'open' in features.columns:
        features['volume_price_trend'] = (features['close'] - features['open']) * features['volume']
        features['close_range_position'] = (features['close'] - features['low']) / (features['high'] - features['low'])
        features['high_low_ratio'] = features['high'] / features['low']
        features['realized_volatility_1m'] = np.log(features['high'] / features['low'])
    
    # 2. Buy/Sell Pressure Features
    if 'buy_volume' in features.columns and 'sell_volume' in features.columns:
        features['buy_sell_ratio'] = features['buy_volume'] / (features['sell_volume'] + 1e-8)
        features['net_volume'] = features['buy_volume'] - features['sell_volume']
    
    # 3. Price and Volume Dynamics
    features['price_change'] = features['close'].diff() if 'close' in features.columns else features['price'].diff()
    features['volume_change'] = features['volume'].diff()
    features['price_acceleration'] = features['price_change'].diff()
    
    # 4. Rolling Statistics
    features['volume_ma_5'] = features['volume'].rolling(5).mean()
    features['volume_ratio'] = features['volume'] / features['volume_ma_5']
    features['price_ma_5'] = features['close'].rolling(5).mean() if 'close' in features.columns else features['price'].rolling(5).mean()
    
    # 5. Tick Direction (Approximation)
    price_col = 'close' if 'close' in features.columns else 'price'
    features['tick_direction'] = np.sign(features[price_col] - features[price_col].shift(1))
    features['cumulative_tick_direction'] = features['tick_direction'].cumsum()
    
    return features


def cost_benefit_analysis(data_size_mb: float, computational_resources: str = 'medium') -> str:
    """
    Análisis costo-beneficio: ¿usar tick data o 1m proxy?
    
    :param data_size_mb: Tamaño de data en MB
    :param computational_resources: 'low', 'medium', 'high'
    :return: Recomendación ('tick_data', '1m_proxy', 'hybrid')
    """
    
    # Ajustar thresholds según recursos computacionales
    thresholds = {
        'low': (50, 200),      # Recursos limitados
        'medium': (100, 1000), # Recursos normales  
        'high': (500, 5000)    # Recursos altos
    }
    
    small_threshold, large_threshold = thresholds.get(computational_resources, thresholds['medium'])
    
    if data_size_mb < small_threshold:
        return "tick_data"  # Vale la pena la precisión extra
    elif data_size_mb < large_threshold:
        return "hybrid"     # Combinar según necesidades
    else:
        return "1m_proxy"   # Costo computacional prohibitivo


def get_1m_proxy_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                            threshold: Optional[float] = None,
                            target_bars_per_day: int = 50,
                            proxy_method: str = 'weighted',
                            batch_size: int = 20000000,
                            verbose: bool = True,
                            to_csv: bool = False,
                            output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Creates dollar bars using 1m proxy data (cost-effective alternative to tick data)
    
    :param file_path_or_df: Path to csv or DataFrame with tick data
    :param threshold: Dollar threshold (if None, auto-calibrates)
    :param target_bars_per_day: Target bars per day for auto-calibration
    :param proxy_method: Method for 1m resampling ('ohlcv', 'weighted', 'last_tick')
    :param batch_size: Number of rows to read per batch
    :param verbose: Print progress messages
    :param to_csv: Save results to CSV
    :param output_path: Output file path if to_csv=True
    :return: Dollar bars from 1m proxy data
    """
    
    # Leer datos
    if isinstance(file_path_or_df, pd.DataFrame):
        raw_data = file_path_or_df
    else:
        raw_data = pd.read_csv(file_path_or_df, nrows=batch_size)
    
    if verbose:
        print(f"Converting tick data to 1m proxy using '{proxy_method}' method...")
    
    # Convertir a 1m proxy
    minute_proxy = resample_to_1min_proxy(raw_data, method=proxy_method)
    
    # Auto-calibrar threshold si no se proporciona
    if threshold is None:
        # Calcular dollar volume para calibración
        price_col = 'close' if 'close' in minute_proxy.columns else 'price'
        minute_proxy['dollar_volume'] = minute_proxy[price_col] * minute_proxy['volume']
        
        daily_dollar_volume = minute_proxy['dollar_volume'].sum()
        # Estimar días (asumiendo 6.5h de trading por día)
        trading_days = len(minute_proxy) / (6.5 * 60)
        threshold = daily_dollar_volume / (target_bars_per_day * trading_days)
        
        if verbose:
            print(f"Auto-calibrated threshold: ${threshold:,.0f} per bar (target: {target_bars_per_day} bars/day)")
    
    # Crear dollar bars usando 1m proxy
    # Preparar datos en formato estándar
    price_col = 'close' if 'close' in minute_proxy.columns else 'price'
    processed_data = minute_proxy[['date_time', price_col, 'volume']].copy()
    processed_data.columns = ['date_time', 'price', 'volume']
    
    # Crear barras
    bars = StandardBars(metric='dollar_bars', threshold=threshold, batch_size=len(processed_data))
    result = bars._extract_bars(processed_data)
    
    if verbose:
        print(f"Generated {len(result)} dollar bars using 1m proxy data")
    
    # Convertir a DataFrame
    if result:
        df_result = pd.DataFrame(result)
        if to_csv and output_path:
            df_result.to_csv(output_path, index=False)
        return df_result
    else:
        return pd.DataFrame()
