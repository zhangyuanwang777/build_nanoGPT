import os
import numpy as np
import logging
from datetime import datetime

class Logger:
    '''用于记录运行过程中的日志信息

    使用日志记录:
    ```python
    logger = Logger()
    logger.mes("开始记录日志信息...", level='info')
    logger.mes("一个调试信息例子。", level='debug')
    logger.mes("一个警告信息例子。", level='warning')
    logger.mes("一个错误信息例子。", level='error')
    logger.mes("一个严重错误信息例子。", level='critical')
    ```

    使用print输出:
    ```python
    logger = Logger(use_logging=False)
    logger.mes("这条信息使用 print 输出。")
    ```

    '''
    def __init__(self, log_directory="logs", use_logging=True) -> None:
        self.log_directory = log_directory
        self.use_logging = use_logging
        if use_logging:
            self._setup_logger()

    def _setup_logger(self) -> None:
        # 创建日志目录
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        
        # 设置日志文件名
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(self.log_directory, f"log_{current_time}.log")

        # 设置日志基本配置
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_filename,
                            filemode='w',
                            datefmt='%m-%d %H:%M:%S')

        # 创建日志记录器
        self.logger = logging.getLogger(__name__)

        # 控制台日志输出设置
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def mes(self, message: str, level: str='info') -> None:
        """
        输出消息到日志或控制台。

        :param message: 要输出的消息
        :param level: 消息级别 ('info', 'debug', 'warning', 'error', 'critical')
        """
        assert level in ['debug', 'info', 'warning', 'error', 'critical'], "Invalid log level."
        if self.use_logging:
            if level == 'debug':
                self.logger.debug(message)
            elif level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'critical':
                self.logger.critical(message)

class VariableTracker:
    '''动态地跟踪和记录程序中的变量'''
    def __init__(self, keys: list, master_process: bool) -> None:
        assert len(keys)>0, "存储变量数不能为0"
        self.keys = keys
        self.len_var = len(keys)
        self.master_process = master_process
        self.variables = {}
        for i in keys:
            self.variables[i] = []
        
    def append(self, name: str, value) -> None:
        """更新现有变量的值"""
        assert name in self.keys, "变量不存在"
        if self.master_process:
            self.variables[name].append(value)

    def save(self, dir_path):
        """保存变量字典到npy文件"""
        np.save(dir_path, self.variables)


if __name__ == "__main__":
    logger = Logger()
    logger.mes("开始记录日志信息...")