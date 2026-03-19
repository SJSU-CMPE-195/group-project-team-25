# **RL-Based Strategies for Improving and Attacking Synthetic CAPTCHAs**

## **Report 4: Revised Implementation Plan**

**CMPE 195B \- Senior Design Project II** **Spring 2026**

---

**Team Members:**

| Name | SJSU Email |
| ----- | ----- |
| Meghana Indukuri | meghana.indukuri@sjsu.edu |
| Eman Naseerkhan | eman.naseerkhan@sjsu.edu |
| Martin Tran | vietnhatminh.tran@sjsu.edu |
| Joshua Rose | joshua.rose@sjsu.edu |

**Project Advisor:** Dr. Younghee Park

**Date:** 2/27/2026

## 

## **1\. Lessons Learned from 195A** 

**What worked well:** Our prototype/proof of concept worked well and helped refine what we had in mind for our project. Team dynamics were great, all members contributed, and the project has been very well structured so far due to our project manager and our allocation of work. **Challenges:** Moving from the initial prototype to actually building the project took longer than expected. This was surprising since after the PoC we felt we had a good idea of the project and its implementation but, we decided later in the semester to change the implementation of the project, which slowed us down. It’s common for project plans to change, especially in software; however, in hindsight, we realize that it would have been useful to determine which project component delays would be the most impactful, so we can plan around them. **Key insight:** One key takeaway from 195A was that implementation details will often cause roadblocks and pivots in the project; thus, we think it may be better in 195B to be more proactive about development so we can encounter these issues earlier (e.g., data collection issues & training model collapse).

## **2\. Revised Approach**

* **Aspect:** Originally planned for the environment to be a desktop application simulating a website; we are now doing a full website because certain aspects of a website are used by the CAPTCHA system, such as user IP and web host.  
* **Scope adjustments:** We added a more modular approach to our project, in which we now have a separate classification model that acts as both a baseline and an improvement mechanism for our RL model. This was done due to the fact that it is important to really understand how a simple model is affected by the data. This modular approach also introduced an offline component to our RL model training.

## **3\. 195B Milestones & Timeline**

* **Prototype Complete** (3/13 — Implementation 2): Key functionality working end-to-end.  
* **RL-Based Captcha** (3/20): CAPTCHA puzzles complete and polished.  
* **Report 5** (3/27 — Report 5): \[Progress update and revised plan.\]  
* **Implementation 3** (4/17 — Implementation 3): Test suite, CI/CD, and deployment ready.  
  * **Stress test** (4/10): Complete testing CAPTCHA with automated bots.   
* **Presentation 2** (4/24 — Presentation 2): Slides and demo rehearsal complete.  
* **Report 6 / Expo** (5/15 — Project Expo): Final polish, poster, and demo ready.