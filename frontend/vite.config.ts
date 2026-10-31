import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({plugins:[react()],build:{sourcemap:false,rolldownOptions:{output:{strictExecutionOrder:true,codeSplitting:{groups:[{name:'charts-vendor',test:/node_modules[\\/](echarts|zrender)/,priority:20,maxSize:350000},{name:'react-vendor',test:/node_modules[\\/](react|react-dom|scheduler)[\\/]/,priority:15},{name:'ui-vendor',test:/node_modules/,priority:10,maxSize:350000}]}}}},server:{host:'127.0.0.1'}});
