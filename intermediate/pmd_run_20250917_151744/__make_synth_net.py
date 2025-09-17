import pandapower as pp, json
from pandapower.file_io import PPJSONEncoder
net = pp.create_empty_network()
b1  = pp.create_bus(net, vn_kv=20.0, name='BUS_HV')
b2  = pp.create_bus(net, vn_kv=0.4, name='BUS_LV')
pp.create_ext_grid(net, bus=b1, vm_pu=1.0, name='ext')
pp.create_transformer_from_parameters(net, hv_bus=b1, lv_bus=b2, sn_mva=0.4, vn_hv_kv=20.0, vn_lv_kv=0.4,
    vk_percent=6.0, vkr_percent=0.5, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0)
pp.create_load(net, bus=b2, p_mw=0.01, q_mvar=0.003, name='LD0')
with open(r'E:\PF_CLEAN\intermediate\pmd_run_20250917_151744\net_power.json','w',encoding='utf-8') as f:
    json.dump(net, f, cls=PPJSONEncoder, indent=2, ensure_ascii=False)
print('OK')
