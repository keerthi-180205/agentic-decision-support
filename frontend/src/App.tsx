import React from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Activity, Database, Terminal, Settings } from 'lucide-react';

export default function App() {
  return (
    <div className="h-screen w-full bg-[#111827] text-[#E5E7EB] flex flex-col font-sans overflow-hidden">
      {/* Top Navbar */}
      <header className="h-12 border-b border-gray-800 bg-[#0D1117] flex items-center px-4 font-semibold shrink-0 shadow-sm z-10">
        <Activity className="w-5 h-5 mr-3 text-indigo-500" />
        <span className="tracking-wide">AI Data Analysis Dashboard</span>
      </header>
      
      {/* Main Workspace */}
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal">
          {/* SIDEBAR PANEL */}
          <Panel defaultSize={20} minSize={10} maxSize={40} className="bg-[#111827] flex flex-col">
            <div className="p-3 font-semibold text-xs text-gray-500 tracking-wider flex items-center shadow-sm">
                <Database className="w-4 h-4 mr-2" />
                EXPLORER
            </div>
            <div className="flex-1 p-4">
               {/* Controls / File Upload */}
               <div className="mb-6">
                 <label className="block text-sm mb-2 text-gray-300 font-medium">Data Ingestion</label>
                 <input type="file" className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 transition-colors cursor-pointer" />
               </div>

               <div className="space-y-2">
                 <div className="text-sm px-2 py-1.5 bg-gray-800/50 rounded text-gray-300 hover:bg-gray-800 cursor-pointer border border-transparent hover:border-gray-700 transition">Configuration</div>
                 <div className="text-sm px-2 py-1.5 rounded text-gray-400 hover:bg-gray-800 cursor-pointer border border-transparent hover:border-gray-700 transition">Model selection</div>
                 <div className="text-sm px-2 py-1.5 rounded text-gray-400 hover:bg-gray-800 cursor-pointer border border-transparent hover:border-gray-700 transition">Logs</div>
               </div>
            </div>
          </Panel>
          
          {/* VERTICAL DRAG HANDLE */}
          <PanelResizeHandle className="w-1 bg-gray-800 hover:bg-indigo-500 active:bg-indigo-400 transition-colors cursor-col-resize group flex items-center justify-center">
            <div className="w-0.5 h-8 bg-gray-600 rounded-full group-hover:bg-white" />
          </PanelResizeHandle>
          
          <Panel>
            <PanelGroup direction="vertical">
              {/* MAIN CONTENT PANEL */}
              <Panel defaultSize={70} minSize={20}>
                <div className="h-full flex flex-col bg-[#0b0f19]">
                  <div className="p-3 font-semibold text-xs text-gray-500 tracking-wider border-b border-gray-800 shadow-sm flex items-center bg-[#111827]">
                    <Settings className="w-4 h-4 mr-2" />
                    MAIN DASHBOARD
                  </div>
                  <div className="flex-1 p-8 overflow-auto">
                    <h2 className="text-2xl font-bold mb-2 text-white">Dataset Summary</h2>
                    <p className="text-gray-400 mb-8">Upload a file to trigger the Agentic Pipeline and run the initial EDA.</p>
                    
                    <div className="border border-gray-800 border-dashed rounded-lg h-64 flex items-center justify-center text-gray-600 bg-gray-800/20">
                      No data loaded in view
                    </div>
                  </div>
                </div>
              </Panel>
              
              {/* HORIZONTAL DRAG HANDLE */}
              <PanelResizeHandle className="h-1 bg-gray-800 hover:bg-indigo-500 active:bg-indigo-400 transition-colors cursor-row-resize group flex items-center justify-center">
                <div className="h-0.5 w-8 bg-gray-600 rounded-full group-hover:bg-white" />
              </PanelResizeHandle>
              
              {/* BOTTOM PANEL (OUTPUT/LOGS) */}
              <Panel defaultSize={30} minSize={10} className="bg-[#090b10] flex flex-col">
                <div className="flex items-center px-4 pt-2 border-b border-gray-800 text-xs font-medium text-gray-500 space-x-6 bg-[#111827]">
                  <div className="cursor-pointer hover:text-gray-200 pb-2">PROBLEMS</div>
                  <div className="cursor-pointer text-indigo-400 border-b-2 border-indigo-500 pb-2 flex items-center">
                     <Terminal className="w-3.5 h-3.5 mr-1" />
                     OUTPUT / LOGS
                  </div>
                  <div className="cursor-pointer hover:text-gray-200 pb-2">DATA PREVIEW</div>
                </div>
                <div className="p-4 font-mono text-sm text-gray-300 overflow-auto h-full space-y-1">
                  <div><span className="text-green-400">[info]</span> Dashboard initialized securely.</div>
                  <div><span className="text-green-400">[info]</span> React resilient panels mounted.</div>
                  <div className="text-gray-500">&gt; Waiting for backend connections...</div>
                </div>
              </Panel>
            </PanelGroup>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}
