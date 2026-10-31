import {createRoot} from 'react-dom/client';
import {ConfigProvider} from 'antd';
import zhCN from 'antd/locale/zh_CN';
import {Single71} from './single71';
import './theme.css';
createRoot(document.getElementById('root')!).render(<ConfigProvider locale={zhCN} theme={{token:{colorPrimary:'#2563eb',colorText:'#172b46',borderRadius:8,fontFamily:'Segoe UI, Microsoft YaHei, sans-serif'}}}><Single71/></ConfigProvider>);
