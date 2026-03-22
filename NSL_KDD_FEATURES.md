# NSL-KDD Feature Reference

## Source
Canadian Institute for Cybersecurity (CIC), UNB  
https://www.unb.ca/cic/datasets/nsl.html

## 41 Features (plus label)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | duration | continuous | Length of connection (seconds) |
| 2 | protocol_type | symbolic | Protocol (tcp, udp, icmp) |
| 3 | service | symbolic | Destination service |
| 4 | flag | symbolic | Connection status |
| 5 | src_bytes | continuous | Bytes from source to destination |
| 6 | dst_bytes | continuous | Bytes from destination to source |
| 7 | land | continuous | 1 if connection is from/to same host/port |
| 8 | wrong_fragment | continuous | Number of wrong fragments |
| 9 | urgent | continuous | Number of urgent packets |
| 10 | hot | continuous | Number of hot indicators |
| 11 | num_failed_logins | continuous | Failed login attempts |
| 12 | logged_in | continuous | 1 if successfully logged in |
| 13 | num_compromised | continuous | Number of compromised conditions |
| 14 | root_shell | continuous | 1 if root shell obtained |
| 15 | su_attempted | continuous | 1 if su root attempted |
| 16 | num_root | continuous | Number of root accesses |
| 17 | num_file_creations | continuous | Number of file creation operations |
| 18 | num_shells | continuous | Number of shell prompts |
| 19 | num_access_files | continuous | Number of operations on access control files |
| 20 | num_outbound_cmds | continuous | Outbound commands in an ftp session |
| 21 | is_host_login | continuous | 1 if login belongs to hot list |
| 22 | is_guest_login | continuous | 1 if guest login |
| 23 | count | continuous | Connections to same host in past 2s |
| 24 | srv_count | continuous | Connections to same service in past 2s |
| 25 | serror_rate | continuous | Connection SYN error rate |
| 26 | srv_serror_rate | continuous | Service SYN error rate |
| 27 | rerror_rate | continuous | Connection REJ error rate |
| 28 | srv_rerror_rate | continuous | Service REJ error rate |
| 29 | same_srv_rate | continuous | Same service rate |
| 30 | diff_srv_rate | continuous | Different service rate |
| 31 | srv_diff_host_rate | continuous | Different host, same service rate |
| 32 | dst_host_count | continuous | Count of connections to destination host |
| 33 | dst_host_srv_count | continuous | Count of connections to same service |
| 34 | dst_host_same_srv_rate | continuous | Same service rate to destination |
| 35 | dst_host_diff_srv_rate | continuous | Different service rate to destination |
| 36 | dst_host_same_src_port_rate | continuous | Same source port rate |
| 37 | dst_host_srv_diff_host_rate | continuous | Different host rate for same service |
| 38 | dst_host_serror_rate | continuous | Destination host SYN error rate |
| 39 | dst_host_srv_serror_rate | continuous | Destination host service SYN error rate |
| 40 | dst_host_rerror_rate | continuous | Destination host REJ error rate |
| 41 | dst_host_srv_rerror_rate | continuous | Destination host service REJ error rate |

## Label (42nd column)
- **normal**: Benign traffic
- Attack types: DoS, Probe, R2L, U2R (various subtypes)

## Constraints
- **Heavy-tailed distributions**: duration, src_bytes, dst_bytes often right-skewed → log-transform
- **Categorical cardinality**: service has many values → encoding strategy matters
- **Zero-inflation**: Many features (e.g., num_root) are mostly 0
