import pandapower as pp, json
from pandapower.file_io import PPJSONEncoder
net = pp.from_pickle(r'E:\PF_CLEAN\saved_json\grids\3b86aa918529\net.p')
with open(r'E:\PF_CLEAN\intermediate\pmd_run_20250917_150636\net_power.json',"w",encoding="utf-8") as f:
    json.dump(net, f, cls=PPJSONEncoder, indent=2, ensure_ascii=False)
print(r'E:\PF_CLEAN\intermediate\pmd_run_20250917_150636\net_power.json')
