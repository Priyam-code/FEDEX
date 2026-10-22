"""Small synthetic fixtures for debugging ONLY; not a replacement for VeReMi."""
import copy
import math
import json
from pathlib import Path
import numpy as np
from .utils import dump_json, fresh_dir


def make_demo(output, seed=31415):
    root = fresh_dir(output)
    variants = ["benign", "constantPositionOffset", "timeDelayAttack", "dosAttack", "reversedHeading"]
    for split_id, split in enumerate(("Train", "Validation", "Test")):
        for variant in (variants if split=="Test" else ["benign"]):
            for receiver in range(12):
                rng = np.random.default_rng(seed+split_id*1000+receiver)
                messages = []
                congested = receiver % 2 == 0
                speed = 7 if congested else 23
                base = split_id*10000+receiver*100
                ox, oy = receiver*1500., receiver % 3*1000.
                for step in range(65):
                    for vehicle in range(5):
                        time = base+step*.1+vehicle*.005
                        v = speed+vehicle*.15+math.sin(step/15+vehicle)*.3
                        pos = [ox+step*.1*speed+vehicle*20+rng.normal(0, .15), oy+(vehicle%2)*3.5+rng.normal(0, .1), 0]
                        sender = dict(pos=pos, spd=v, acl=math.cos(step/15+vehicle)*.2,
                                      hed=90+rng.normal(0, .5))
                        msg = dict(rcvTime=int((time+.002+abs(rng.normal(0, .0002)))*1e9),
                                   sendTime=int(time*1e9), sender_id=f"veh_{vehicle}",
                                   sender_alias=str(receiver*100+vehicle), messageID=step*10+vehicle,
                                   receiver=dict(pos=[ox+step*.1*speed-10, oy, 0]), sender=sender)
                        attacked = variant != "benign" and vehicle < 2 and step >= 30
                        if variant != "benign":
                            msg["attacker"] = int(attacked)
                        # Baseline intentionally has no attacker key: matches official base archives.
                        if attacked:
                            if variant == "constantPositionOffset":
                                msg["sender"]["pos"][0] += 180
                            elif variant == "timeDelayAttack":
                                msg["rcvTime"] += int(.8e9)
                            elif variant == "reversedHeading":
                                msg["sender"]["hed"] = (msg["sender"]["hed"]+180) % 360
                        messages.append(msg)
                        if attacked and variant == "dosAttack":
                            for duplicate in range(1, 5):
                                replay = copy.deepcopy(msg)
                                replay["rcvTime"] += duplicate*100_000
                                messages.append(replay)
                path = root/("benign" if variant=="benign" else "attacks")/variant/split/f"receiver_{receiver}.json"
                dump_json(path, sorted(messages, key=lambda m: m["rcvTime"]))
    dump_json(root/"manifest.json", dict(origin="synthetic_demo", timestamp_unit="ns", sources=[
        dict(path="benign", role="baseline", assume_benign=True, scenario="synthetic"),
        dict(path="attacks", role="attack", assume_benign=False, scenario="synthetic")],
        warning="Generated toy trajectories and obvious attacks. NEVER claim these as VeReMi or publication performance."))
    return root/"manifest.json"
