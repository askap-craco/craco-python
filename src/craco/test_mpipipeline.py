from craco import mpipipeline

def test_parse_host_devices():
    hosts = [f'seren-{i:02d}' for i in range(1,11)]
    devstr = 'seren-01:1,seren-04:0,seren-05:0-1'
    devices = (0,1)
    host_devices = mpipipeline.parse_host_devices(hosts, devstr, devices)
    assert len(host_devices) == len(hosts)
    for h in hosts:
        devs = host_devices[h]
        if h == 'seren-01':
            assert devs == (0,)
        elif h == 'seren-04':
            assert devs == (1,)
        elif h == 'seren-05':
            assert devs == ()
        else:
            assert devs == (0,1)
            
