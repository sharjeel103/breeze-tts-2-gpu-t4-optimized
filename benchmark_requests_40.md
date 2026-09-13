# 40 Benchmark Requests: Full Dataset & Empirical QoS Performance

**Test Run**: 40-Request Balanced Burst Stress Test  
**Architecture**: Design 5 — Dual Independent Assembly Stations (`cuda:0` / `cuda:1`) + Dedicated Streaming Vocoder (`cuda:1`)  
**Hardware**: Dual NVIDIA Tesla T4 (16GB GDDR6 each, sm_75)  
**Overall Cluster Throughput**: **1.38x Real-Time** (288.25s of speech produced in 208.16s wall-clock time)  
**Cluster Saturation**: GPU 0: 86.3% util (65.4W / 70W) | GPU 1: 94.7% util (65.6W / 70W)  

---

## 1. Complete Request Registry & Prompts

| ID | Category | Tone / Instruction | Prompt Text |
| :--- | :--- | :--- | :--- |
| **`Req_01`** | Customer Service & E-Commerce | *Clean and polite.* | Your order has been confirmed and is currently being packed for express delivery this afternoon. |
| **`Req_02`** | IT & System Updates | *Calm female assistant.* | The system update completed successfully without any errors, and all security patches have been safely installed. |
| **`Req_03`** | Personal Assistant & Scheduling | *Friendly customer service.* | Good morning! Please let me know how I can assist you with your schedule and appointments today. |
| **`Req_04`** | Transit & Travel | *Airport announcement.* | Flight seven twelve to San Francisco is now boarding at terminal two, gate twenty-four. |
| **`Req_05`** | Enterprise & Calendar | *Professional assistant.* | The project planning meeting has been rescheduled to Thursday at three in the afternoon, conference room four. |
| **`Req_06`** | Fintech & Billing | *Polite confirmation.* | Your payment was processed successfully, and an itemized digital receipt has been sent to your primary email. |
| **`Req_07`** | Broadcast & Weather | *Radio broadcaster.* | The air quality index is currently moderate across the valley with light northwesterly winds throughout the morning. |
| **`Req_08`** | Network & Cloud Operations | *Technical narrator.* | All cluster network services are running normally with optimal bandwidth and zero packet loss detected. |
| **`Req_09`** | Team Workflow & Productivity | *Courteous reminder.* | Please remember to submit your weekly engineering progress report before five this evening for team review. |
| **`Req_10`** | Rideshare & Navigation | *Navigation prompt.* | Your ride has arrived outside the main hotel lobby. The silver vehicle license plate is five alpha seven. |
| **`Req_11`** | Weather & Environment | *Weather reporter.* | Today will be mostly clear and sunny with mild afternoon temperatures reaching seventy-four degrees across the city. |
| **`Req_12`** | Events & Hospitality | *Event coordinator.* | The conference keynote begins in ten minutes in the primary auditorium on floor three, open to all attendees. |
| **`Req_13`** | Healthcare & Pharmacy | *Healthcare assistant.* | Your prescription order is ready for pickup at the neighborhood pharmacy counter on Maple Avenue. |
| **`Req_14`** | Rail & Commuter Transit | *Transit announcement.* | The express commuter train to central station will depart from platform four in exactly six minutes. |
| **`Req_15`** | Smart Home & IoT | *Helpful smart home assistant.* | A new firmware update is available for your smart display. Please ensure a stable Wi-Fi connection to proceed. |
| **`Req_16`** | Banking & Financial | *Banking alert.* | Your checking account balance has been updated following the recent automated monthly savings transfer. |
| **`Req_17`** | Campus & Education | *Campus announcement.* | The university library will be closing in fifteen minutes. Please bring all borrowed materials to the front circulation desk. |
| **`Req_18`** | Traffic & Navigation | *Navigation assistant.* | Traffic on the interstate highway is moving smoothly with an estimated total travel time of twenty-two minutes. |
| **`Req_19`** | Museum & Tourism | *Tour guide.* | Welcome to the national science center. Guided audio tours commence every hour on the hour at the main rotunda. |
| **`Req_20`** | Hospitality & Dining | *Concierge tone.* | Your table reservation for four guests at Bistro Bella has been confirmed for eight tonight on the patio. |
| **`Req_21`** | Logistics & Delivery | *Office concierge.* | The morning courier package has been safely delivered to the front reception desk for your immediate collection. |
| **`Req_22`** | IT Infrastructure | *System administrator.* | Routine server infrastructure maintenance is scheduled for tonight at midnight and will last approximately one hour. |
| **`Req_23`** | Nightly Forecast | *Calm narrator.* | Temperatures will drop noticeably tonight under clear starry skies with a gentle autumn breeze from the north. |
| **`Req_24`** | Aviation & Travel | *Airline assistant.* | Your international flight check-in is complete, and your digital boarding passes have been synchronized to your phone. |
| **`Req_25`** | Virtual Events & Webinars | *Webinar host.* | The live technical webinar on distributed computing architectures will begin promptly at noon Eastern Standard Time. |
| **`Req_26`** | Cybersecurity & Alerts | *Security alert.* | Security notification: a new login was detected from a personal laptop in Chicago, Illinois. Please verify your identity. |
| **`Req_27`** | Facilities & Property | *Building announcement.* | The passenger elevator on the north wing is currently undergoing maintenance and will reopen at two this afternoon. |
| **`Req_28`** | SaaS & Subscriptions | *Customer care.* | Your premium software subscription has been renewed successfully, unlocking continuous priority access to all cloud tools. |
| **`Req_29`** | Airport Shuttle & Transit | *Transit audio.* | Passengers traveling to terminal B should proceed to shuttle stop three for immediate baggage transfer. |
| **`Req_30`** | Urban Mobility | *City transit guide.* | The downtown business shuttle departs every fifteen minutes from the central transit plaza near the historic clock tower. |
| **`Req_31`** | 2FA & Identity | *Verification voice.* | A temporary authorization code has been dispatched to your mobile phone number via secure text messaging. |
| **`Req_32`** | Hospitality & Amenities | *Hospitality host.* | The resident fitness facility will remain open until eleven tonight for all registered hotel and club members. |
| **`Req_33`** | Traffic Monitoring | *Traffic broadcast.* | Local traffic monitors report minor road construction delays near the east river crossing during evening peak hours. |
| **`Req_34`** | Business Intelligence | *Business assistant.* | Your analytical quarterly summary report has finished generating and is now available for download on the management portal. |
| **`Req_35`** | Academic & Labs | *Instructor voice.* | The interactive workshop on modern deep learning frameworks begins at ten sharp in computer laboratory C. |
| **`Req_36`** | Retail & Guest Services | *Warm goodbye.* | Thank you for visiting our technology showroom today. Please take your complimentary catalog and have a wonderful day. |
| **`Req_37`** | Weather Forecasting | *Weather anchor.* | Tomorrow's weather forecast calls for brief morning showers followed by pleasant sunshine and light southerly breezes. |
| **`Req_38`** | Medical & Clinical | *Medical receptionist.* | Your consultation appointment with Doctor Reynolds has been confirmed for Tuesday morning at ten thirty. |
| **`Req_39`** | DevOps & SRE | *DevOps assistant.* | The production cluster deployment completed without incident, and all containerized microservices report healthy operational status. |
| **`Req_40`** | Wealth & Banking | *Financial adviser.* | All pending banking transactions have cleared, and your comprehensive monthly financial statement is now available to view. |

---

## 2. Empirical QoS Breakdown per Request

| ID | Station | Device | Audio (s) | Queue Wait (s) | Compute (s) | TTFA (ms) | Active RTF | Turnaround RTF | Exit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Req_01` | **A** | `cuda:0` | 5.53s | 0.21s | 8.41s | **261.7 ms** | 1.52x | 1.56x | T+8.62s |
| `Req_02` | **B** | `cuda:1` | 6.97s | 0.47s | 10.38s | **310.0 ms** | 1.49x | 1.56x | T+10.85s |
| `Req_03` | **A** | `cuda:0` | 5.37s | 8.48s | 8.04s | **249.0 ms** | 1.50x | 3.08x | T+16.52s |
| `Req_04` | **B** | `cuda:1` | 5.05s | 10.72s | 7.30s | **242.9 ms** | 1.45x | 3.57x | T+18.02s |
| `Req_05` | **A** | `cuda:0` | 7.21s | 15.85s | 10.24s | **673.3 ms** | 1.42x | 3.62x | T+26.09s |
| `Req_06` | **B** | `cuda:1` | 6.33s | 17.88s | 8.75s | **266.2 ms** | 1.38x | 4.21x | T+26.63s |
| `Req_07` | **A** | `cuda:0` | 6.65s | 25.96s | 9.17s | **238.2 ms** | 1.38x | 5.28x | T+35.13s |
| `Req_08` | **B** | `cuda:1` | 5.85s | 26.51s | 8.84s | **235.0 ms** | 1.51x | 6.04x | T+35.34s |
| `Req_09` | **A** | `cuda:0` | 6.89s | 35.01s | 9.53s | **230.3 ms** | 1.38x | 6.46x | T+44.54s |
| `Req_10` | **B** | `cuda:1` | 7.05s | 35.34s | 9.76s | **214.2 ms** | 1.38x | 6.40x | T+45.11s |
| `Req_11` | **A** | `cuda:0` | 5.93s | 44.42s | 8.35s | **230.4 ms** | 1.41x | 8.90x | T+52.78s |
| `Req_12` | **B** | `cuda:1` | 9.13s | 44.96s | 12.78s | **253.5 ms** | 1.40x | 6.32x | T+57.74s |
| `Req_13` | **A** | `cuda:0` | 8.25s | 52.65s | 11.65s | **239.6 ms** | 1.41x | 7.79x | T+64.30s |
| `Req_14` | **B** | `cuda:1` | 5.53s | 57.61s | 7.93s | **241.1 ms** | 1.44x | 11.86x | T+65.55s |
| `Req_15` | **A** | `cuda:0` | 6.01s | 64.16s | 8.74s | **252.0 ms** | 1.45x | 12.13x | T+72.90s |
| `Req_16` | **B** | `cuda:1` | 5.53s | 65.41s | 8.07s | **253.8 ms** | 1.46x | 13.29x | T+73.48s |
| `Req_17` | **A** | `cuda:0` | 7.05s | 72.72s | 10.34s | **294.1 ms** | 1.47x | 11.78x | T+83.06s |
| `Req_18` | **B** | `cuda:1` | 6.09s | 73.33s | 8.91s | **259.5 ms** | 1.46x | 13.51x | T+82.25s |
| `Req_19` | **B** | `cuda:1` | 7.21s | 82.11s | 10.59s | **256.0 ms** | 1.47x | 12.85x | T+92.69s |
| `Req_20` | **A** | `cuda:0` | 9.05s | 82.91s | 13.97s | **265.1 ms** | 1.54x | 10.70x | T+96.88s |
| `Req_21` | **B** | `cuda:1` | 5.21s | 92.56s | 8.36s | **248.6 ms** | 1.60x | 19.38x | T+100.91s |
| `Req_22` | **A** | `cuda:0` | 5.77s | 96.88s | 8.29s | **220.8 ms** | 1.44x | 18.23x | T+105.17s |
| `Req_23` | **B** | `cuda:1` | 9.21s | 100.92s | 13.07s | **221.5 ms** | 1.42x | 12.37x | T+113.98s |
| `Req_24` | **A** | `cuda:0` | 13.78s | 105.05s | 19.33s | **242.3 ms** | 1.40x | 9.03x | T+124.37s |
| `Req_25` | **B** | `cuda:1` | 7.45s | 113.85s | 10.52s | **244.0 ms** | 1.41x | 16.69x | T+124.37s |
| `Req_26` | **A** | `cuda:0` | 9.61s | 124.19s | 13.67s | **303.2 ms** | 1.42x | 14.34x | T+137.86s |
| `Req_27` | **B** | `cuda:1` | 8.73s | 124.22s | 12.39s | **271.0 ms** | 1.42x | 15.64x | T+136.61s |
| `Req_28` | **B** | `cuda:1` | 7.93s | 136.46s | 11.37s | **253.5 ms** | 1.43x | 18.64x | T+147.83s |
| `Req_29` | **A** | `cuda:0` | 8.33s | 137.72s | 11.89s | **251.1 ms** | 1.43x | 17.96x | T+149.61s |
| `Req_30` | **B** | `cuda:1` | 7.69s | 147.65s | 11.12s | **292.7 ms** | 1.45x | 20.64x | T+158.78s |
| `Req_31` | **A** | `cuda:0` | 7.69s | 149.48s | 11.09s | **262.2 ms** | 1.44x | 20.88x | T+160.57s |
| `Req_32` | **B** | `cuda:1` | 7.61s | 158.58s | 11.02s | **308.8 ms** | 1.45x | 22.28x | T+169.60s |
| `Req_33` | **A** | `cuda:0` | 7.61s | 160.43s | 10.99s | **275.8 ms** | 1.44x | 22.52x | T+171.42s |
| `Req_34` | **B** | `cuda:1` | 7.77s | 169.45s | 11.94s | **281.9 ms** | 1.54x | 23.34x | T+181.39s |
| `Req_35` | **A** | `cuda:0` | 8.73s | 171.28s | 12.47s | **272.3 ms** | 1.43x | 21.04x | T+183.76s |
| `Req_36` | **B** | `cuda:1` | 6.65s | 181.39s | 9.53s | **226.1 ms** | 1.43x | 28.71x | T+190.93s |
| `Req_37` | **A** | `cuda:0` | 5.61s | 183.62s | 8.11s | **248.4 ms** | 1.45x | 34.19x | T+191.73s |
| `Req_38` | **B** | `cuda:1` | 5.37s | 190.79s | 7.77s | **271.6 ms** | 1.45x | 36.99x | T+198.56s |
| `Req_39` | **A** | `cuda:0` | 7.77s | 191.59s | 11.87s | **242.3 ms** | 1.53x | 26.18x | T+203.47s |
| `Req_40` | **B** | `cuda:1` | 7.05s | 198.43s | 9.73s | **242.6 ms** | 1.38x | 29.53x | T+208.16s |

---

## 3. Detailed Request Cards

### `Req_01` — Customer Service & E-Commerce

- **Prompt**: "Your order has been confirmed and is currently being packed for express delivery this afternoon."
- **Voice Style**: *Clean and polite.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `5.53s`
- **Post-Admit TTFA**: `261.7 ms`
- **Active Synthesis Duration**: `8.41s` (Active RTF: `1.52x`)
- **Queue Wait**: `0.21s` (Turnaround RTF: `1.56x`)
- **Stream Completion**: `T+8.62s`

### `Req_02` — IT & System Updates

- **Prompt**: "The system update completed successfully without any errors, and all security patches have been safely installed."
- **Voice Style**: *Calm female assistant.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `6.97s`
- **Post-Admit TTFA**: `310.0 ms`
- **Active Synthesis Duration**: `10.38s` (Active RTF: `1.49x`)
- **Queue Wait**: `0.47s` (Turnaround RTF: `1.56x`)
- **Stream Completion**: `T+10.85s`

### `Req_03` — Personal Assistant & Scheduling

- **Prompt**: "Good morning! Please let me know how I can assist you with your schedule and appointments today."
- **Voice Style**: *Friendly customer service.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `5.37s`
- **Post-Admit TTFA**: `249.0 ms`
- **Active Synthesis Duration**: `8.04s` (Active RTF: `1.50x`)
- **Queue Wait**: `8.48s` (Turnaround RTF: `3.08x`)
- **Stream Completion**: `T+16.52s`

### `Req_04` — Transit & Travel

- **Prompt**: "Flight seven twelve to San Francisco is now boarding at terminal two, gate twenty-four."
- **Voice Style**: *Airport announcement.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.05s`
- **Post-Admit TTFA**: `242.9 ms`
- **Active Synthesis Duration**: `7.30s` (Active RTF: `1.45x`)
- **Queue Wait**: `10.72s` (Turnaround RTF: `3.57x`)
- **Stream Completion**: `T+18.02s`

### `Req_05` — Enterprise & Calendar

- **Prompt**: "The project planning meeting has been rescheduled to Thursday at three in the afternoon, conference room four."
- **Voice Style**: *Professional assistant.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `7.21s`
- **Post-Admit TTFA**: `673.3 ms`
- **Active Synthesis Duration**: `10.24s` (Active RTF: `1.42x`)
- **Queue Wait**: `15.85s` (Turnaround RTF: `3.62x`)
- **Stream Completion**: `T+26.09s`

### `Req_06` — Fintech & Billing

- **Prompt**: "Your payment was processed successfully, and an itemized digital receipt has been sent to your primary email."
- **Voice Style**: *Polite confirmation.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `6.33s`
- **Post-Admit TTFA**: `266.2 ms`
- **Active Synthesis Duration**: `8.75s` (Active RTF: `1.38x`)
- **Queue Wait**: `17.88s` (Turnaround RTF: `4.21x`)
- **Stream Completion**: `T+26.63s`

### `Req_07` — Broadcast & Weather

- **Prompt**: "The air quality index is currently moderate across the valley with light northwesterly winds throughout the morning."
- **Voice Style**: *Radio broadcaster.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `6.65s`
- **Post-Admit TTFA**: `238.2 ms`
- **Active Synthesis Duration**: `9.17s` (Active RTF: `1.38x`)
- **Queue Wait**: `25.96s` (Turnaround RTF: `5.28x`)
- **Stream Completion**: `T+35.13s`

### `Req_08` — Network & Cloud Operations

- **Prompt**: "All cluster network services are running normally with optimal bandwidth and zero packet loss detected."
- **Voice Style**: *Technical narrator.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.85s`
- **Post-Admit TTFA**: `235.0 ms`
- **Active Synthesis Duration**: `8.84s` (Active RTF: `1.51x`)
- **Queue Wait**: `26.51s` (Turnaround RTF: `6.04x`)
- **Stream Completion**: `T+35.34s`

### `Req_09` — Team Workflow & Productivity

- **Prompt**: "Please remember to submit your weekly engineering progress report before five this evening for team review."
- **Voice Style**: *Courteous reminder.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `6.89s`
- **Post-Admit TTFA**: `230.3 ms`
- **Active Synthesis Duration**: `9.53s` (Active RTF: `1.38x`)
- **Queue Wait**: `35.01s` (Turnaround RTF: `6.46x`)
- **Stream Completion**: `T+44.54s`

### `Req_10` — Rideshare & Navigation

- **Prompt**: "Your ride has arrived outside the main hotel lobby. The silver vehicle license plate is five alpha seven."
- **Voice Style**: *Navigation prompt.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.05s`
- **Post-Admit TTFA**: `214.2 ms`
- **Active Synthesis Duration**: `9.76s` (Active RTF: `1.38x`)
- **Queue Wait**: `35.34s` (Turnaround RTF: `6.40x`)
- **Stream Completion**: `T+45.11s`

### `Req_11` — Weather & Environment

- **Prompt**: "Today will be mostly clear and sunny with mild afternoon temperatures reaching seventy-four degrees across the city."
- **Voice Style**: *Weather reporter.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `5.93s`
- **Post-Admit TTFA**: `230.4 ms`
- **Active Synthesis Duration**: `8.35s` (Active RTF: `1.41x`)
- **Queue Wait**: `44.42s` (Turnaround RTF: `8.90x`)
- **Stream Completion**: `T+52.78s`

### `Req_12` — Events & Hospitality

- **Prompt**: "The conference keynote begins in ten minutes in the primary auditorium on floor three, open to all attendees."
- **Voice Style**: *Event coordinator.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `9.13s`
- **Post-Admit TTFA**: `253.5 ms`
- **Active Synthesis Duration**: `12.78s` (Active RTF: `1.40x`)
- **Queue Wait**: `44.96s` (Turnaround RTF: `6.32x`)
- **Stream Completion**: `T+57.74s`

### `Req_13` — Healthcare & Pharmacy

- **Prompt**: "Your prescription order is ready for pickup at the neighborhood pharmacy counter on Maple Avenue."
- **Voice Style**: *Healthcare assistant.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `8.25s`
- **Post-Admit TTFA**: `239.6 ms`
- **Active Synthesis Duration**: `11.65s` (Active RTF: `1.41x`)
- **Queue Wait**: `52.65s` (Turnaround RTF: `7.79x`)
- **Stream Completion**: `T+64.30s`

### `Req_14` — Rail & Commuter Transit

- **Prompt**: "The express commuter train to central station will depart from platform four in exactly six minutes."
- **Voice Style**: *Transit announcement.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.53s`
- **Post-Admit TTFA**: `241.1 ms`
- **Active Synthesis Duration**: `7.93s` (Active RTF: `1.44x`)
- **Queue Wait**: `57.61s` (Turnaround RTF: `11.86x`)
- **Stream Completion**: `T+65.55s`

### `Req_15` — Smart Home & IoT

- **Prompt**: "A new firmware update is available for your smart display. Please ensure a stable Wi-Fi connection to proceed."
- **Voice Style**: *Helpful smart home assistant.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `6.01s`
- **Post-Admit TTFA**: `252.0 ms`
- **Active Synthesis Duration**: `8.74s` (Active RTF: `1.45x`)
- **Queue Wait**: `64.16s` (Turnaround RTF: `12.13x`)
- **Stream Completion**: `T+72.90s`

### `Req_16` — Banking & Financial

- **Prompt**: "Your checking account balance has been updated following the recent automated monthly savings transfer."
- **Voice Style**: *Banking alert.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.53s`
- **Post-Admit TTFA**: `253.8 ms`
- **Active Synthesis Duration**: `8.07s` (Active RTF: `1.46x`)
- **Queue Wait**: `65.41s` (Turnaround RTF: `13.29x`)
- **Stream Completion**: `T+73.48s`

### `Req_17` — Campus & Education

- **Prompt**: "The university library will be closing in fifteen minutes. Please bring all borrowed materials to the front circulation desk."
- **Voice Style**: *Campus announcement.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `7.05s`
- **Post-Admit TTFA**: `294.1 ms`
- **Active Synthesis Duration**: `10.34s` (Active RTF: `1.47x`)
- **Queue Wait**: `72.72s` (Turnaround RTF: `11.78x`)
- **Stream Completion**: `T+83.06s`

### `Req_18` — Traffic & Navigation

- **Prompt**: "Traffic on the interstate highway is moving smoothly with an estimated total travel time of twenty-two minutes."
- **Voice Style**: *Navigation assistant.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `6.09s`
- **Post-Admit TTFA**: `259.5 ms`
- **Active Synthesis Duration**: `8.91s` (Active RTF: `1.46x`)
- **Queue Wait**: `73.33s` (Turnaround RTF: `13.51x`)
- **Stream Completion**: `T+82.25s`

### `Req_19` — Museum & Tourism

- **Prompt**: "Welcome to the national science center. Guided audio tours commence every hour on the hour at the main rotunda."
- **Voice Style**: *Tour guide.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.21s`
- **Post-Admit TTFA**: `256.0 ms`
- **Active Synthesis Duration**: `10.59s` (Active RTF: `1.47x`)
- **Queue Wait**: `82.11s` (Turnaround RTF: `12.85x`)
- **Stream Completion**: `T+92.69s`

### `Req_20` — Hospitality & Dining

- **Prompt**: "Your table reservation for four guests at Bistro Bella has been confirmed for eight tonight on the patio."
- **Voice Style**: *Concierge tone.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `9.05s`
- **Post-Admit TTFA**: `265.1 ms`
- **Active Synthesis Duration**: `13.97s` (Active RTF: `1.54x`)
- **Queue Wait**: `82.91s` (Turnaround RTF: `10.70x`)
- **Stream Completion**: `T+96.88s`

### `Req_21` — Logistics & Delivery

- **Prompt**: "The morning courier package has been safely delivered to the front reception desk for your immediate collection."
- **Voice Style**: *Office concierge.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.21s`
- **Post-Admit TTFA**: `248.6 ms`
- **Active Synthesis Duration**: `8.36s` (Active RTF: `1.60x`)
- **Queue Wait**: `92.56s` (Turnaround RTF: `19.38x`)
- **Stream Completion**: `T+100.91s`

### `Req_22` — IT Infrastructure

- **Prompt**: "Routine server infrastructure maintenance is scheduled for tonight at midnight and will last approximately one hour."
- **Voice Style**: *System administrator.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `5.77s`
- **Post-Admit TTFA**: `220.8 ms`
- **Active Synthesis Duration**: `8.29s` (Active RTF: `1.44x`)
- **Queue Wait**: `96.88s` (Turnaround RTF: `18.23x`)
- **Stream Completion**: `T+105.17s`

### `Req_23` — Nightly Forecast

- **Prompt**: "Temperatures will drop noticeably tonight under clear starry skies with a gentle autumn breeze from the north."
- **Voice Style**: *Calm narrator.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `9.21s`
- **Post-Admit TTFA**: `221.5 ms`
- **Active Synthesis Duration**: `13.07s` (Active RTF: `1.42x`)
- **Queue Wait**: `100.92s` (Turnaround RTF: `12.37x`)
- **Stream Completion**: `T+113.98s`

### `Req_24` — Aviation & Travel

- **Prompt**: "Your international flight check-in is complete, and your digital boarding passes have been synchronized to your phone."
- **Voice Style**: *Airline assistant.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `13.78s`
- **Post-Admit TTFA**: `242.3 ms`
- **Active Synthesis Duration**: `19.33s` (Active RTF: `1.40x`)
- **Queue Wait**: `105.05s` (Turnaround RTF: `9.03x`)
- **Stream Completion**: `T+124.37s`

### `Req_25` — Virtual Events & Webinars

- **Prompt**: "The live technical webinar on distributed computing architectures will begin promptly at noon Eastern Standard Time."
- **Voice Style**: *Webinar host.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.45s`
- **Post-Admit TTFA**: `244.0 ms`
- **Active Synthesis Duration**: `10.52s` (Active RTF: `1.41x`)
- **Queue Wait**: `113.85s` (Turnaround RTF: `16.69x`)
- **Stream Completion**: `T+124.37s`

### `Req_26` — Cybersecurity & Alerts

- **Prompt**: "Security notification: a new login was detected from a personal laptop in Chicago, Illinois. Please verify your identity."
- **Voice Style**: *Security alert.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `9.61s`
- **Post-Admit TTFA**: `303.2 ms`
- **Active Synthesis Duration**: `13.67s` (Active RTF: `1.42x`)
- **Queue Wait**: `124.19s` (Turnaround RTF: `14.34x`)
- **Stream Completion**: `T+137.86s`

### `Req_27` — Facilities & Property

- **Prompt**: "The passenger elevator on the north wing is currently undergoing maintenance and will reopen at two this afternoon."
- **Voice Style**: *Building announcement.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `8.73s`
- **Post-Admit TTFA**: `271.0 ms`
- **Active Synthesis Duration**: `12.39s` (Active RTF: `1.42x`)
- **Queue Wait**: `124.22s` (Turnaround RTF: `15.64x`)
- **Stream Completion**: `T+136.61s`

### `Req_28` — SaaS & Subscriptions

- **Prompt**: "Your premium software subscription has been renewed successfully, unlocking continuous priority access to all cloud tools."
- **Voice Style**: *Customer care.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.93s`
- **Post-Admit TTFA**: `253.5 ms`
- **Active Synthesis Duration**: `11.37s` (Active RTF: `1.43x`)
- **Queue Wait**: `136.46s` (Turnaround RTF: `18.64x`)
- **Stream Completion**: `T+147.83s`

### `Req_29` — Airport Shuttle & Transit

- **Prompt**: "Passengers traveling to terminal B should proceed to shuttle stop three for immediate baggage transfer."
- **Voice Style**: *Transit audio.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `8.33s`
- **Post-Admit TTFA**: `251.1 ms`
- **Active Synthesis Duration**: `11.89s` (Active RTF: `1.43x`)
- **Queue Wait**: `137.72s` (Turnaround RTF: `17.96x`)
- **Stream Completion**: `T+149.61s`

### `Req_30` — Urban Mobility

- **Prompt**: "The downtown business shuttle departs every fifteen minutes from the central transit plaza near the historic clock tower."
- **Voice Style**: *City transit guide.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.69s`
- **Post-Admit TTFA**: `292.7 ms`
- **Active Synthesis Duration**: `11.12s` (Active RTF: `1.45x`)
- **Queue Wait**: `147.65s` (Turnaround RTF: `20.64x`)
- **Stream Completion**: `T+158.78s`

### `Req_31` — 2FA & Identity

- **Prompt**: "A temporary authorization code has been dispatched to your mobile phone number via secure text messaging."
- **Voice Style**: *Verification voice.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `7.69s`
- **Post-Admit TTFA**: `262.2 ms`
- **Active Synthesis Duration**: `11.09s` (Active RTF: `1.44x`)
- **Queue Wait**: `149.48s` (Turnaround RTF: `20.88x`)
- **Stream Completion**: `T+160.57s`

### `Req_32` — Hospitality & Amenities

- **Prompt**: "The resident fitness facility will remain open until eleven tonight for all registered hotel and club members."
- **Voice Style**: *Hospitality host.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.61s`
- **Post-Admit TTFA**: `308.8 ms`
- **Active Synthesis Duration**: `11.02s` (Active RTF: `1.45x`)
- **Queue Wait**: `158.58s` (Turnaround RTF: `22.28x`)
- **Stream Completion**: `T+169.60s`

### `Req_33` — Traffic Monitoring

- **Prompt**: "Local traffic monitors report minor road construction delays near the east river crossing during evening peak hours."
- **Voice Style**: *Traffic broadcast.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `7.61s`
- **Post-Admit TTFA**: `275.8 ms`
- **Active Synthesis Duration**: `10.99s` (Active RTF: `1.44x`)
- **Queue Wait**: `160.43s` (Turnaround RTF: `22.52x`)
- **Stream Completion**: `T+171.42s`

### `Req_34` — Business Intelligence

- **Prompt**: "Your analytical quarterly summary report has finished generating and is now available for download on the management portal."
- **Voice Style**: *Business assistant.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.77s`
- **Post-Admit TTFA**: `281.9 ms`
- **Active Synthesis Duration**: `11.94s` (Active RTF: `1.54x`)
- **Queue Wait**: `169.45s` (Turnaround RTF: `23.34x`)
- **Stream Completion**: `T+181.39s`

### `Req_35` — Academic & Labs

- **Prompt**: "The interactive workshop on modern deep learning frameworks begins at ten sharp in computer laboratory C."
- **Voice Style**: *Instructor voice.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `8.73s`
- **Post-Admit TTFA**: `272.3 ms`
- **Active Synthesis Duration**: `12.47s` (Active RTF: `1.43x`)
- **Queue Wait**: `171.28s` (Turnaround RTF: `21.04x`)
- **Stream Completion**: `T+183.76s`

### `Req_36` — Retail & Guest Services

- **Prompt**: "Thank you for visiting our technology showroom today. Please take your complimentary catalog and have a wonderful day."
- **Voice Style**: *Warm goodbye.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `6.65s`
- **Post-Admit TTFA**: `226.1 ms`
- **Active Synthesis Duration**: `9.53s` (Active RTF: `1.43x`)
- **Queue Wait**: `181.39s` (Turnaround RTF: `28.71x`)
- **Stream Completion**: `T+190.93s`

### `Req_37` — Weather Forecasting

- **Prompt**: "Tomorrow's weather forecast calls for brief morning showers followed by pleasant sunshine and light southerly breezes."
- **Voice Style**: *Weather anchor.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `5.61s`
- **Post-Admit TTFA**: `248.4 ms`
- **Active Synthesis Duration**: `8.11s` (Active RTF: `1.45x`)
- **Queue Wait**: `183.62s` (Turnaround RTF: `34.19x`)
- **Stream Completion**: `T+191.73s`

### `Req_38` — Medical & Clinical

- **Prompt**: "Your consultation appointment with Doctor Reynolds has been confirmed for Tuesday morning at ten thirty."
- **Voice Style**: *Medical receptionist.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `5.37s`
- **Post-Admit TTFA**: `271.6 ms`
- **Active Synthesis Duration**: `7.77s` (Active RTF: `1.45x`)
- **Queue Wait**: `190.79s` (Turnaround RTF: `36.99x`)
- **Stream Completion**: `T+198.56s`

### `Req_39` — DevOps & SRE

- **Prompt**: "The production cluster deployment completed without incident, and all containerized microservices report healthy operational status."
- **Voice Style**: *DevOps assistant.*
- **Assigned Worker**: Station **A** on `cuda:0`
- **Audio Generated**: `7.77s`
- **Post-Admit TTFA**: `242.3 ms`
- **Active Synthesis Duration**: `11.87s` (Active RTF: `1.53x`)
- **Queue Wait**: `191.59s` (Turnaround RTF: `26.18x`)
- **Stream Completion**: `T+203.47s`

### `Req_40` — Wealth & Banking

- **Prompt**: "All pending banking transactions have cleared, and your comprehensive monthly financial statement is now available to view."
- **Voice Style**: *Financial adviser.*
- **Assigned Worker**: Station **B** on `cuda:1`
- **Audio Generated**: `7.05s`
- **Post-Admit TTFA**: `242.6 ms`
- **Active Synthesis Duration**: `9.73s` (Active RTF: `1.38x`)
- **Queue Wait**: `198.43s` (Turnaround RTF: `29.53x`)
- **Stream Completion**: `T+208.16s`

