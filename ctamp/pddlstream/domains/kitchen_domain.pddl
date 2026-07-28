; Adapted from caelan/pddlstream examples/kitchen/domain.pddl (kitchen2d).
; Keeps its gripper, pose, grasp, pick, and place vocabulary. Benchmark-only
; clean and cook actions replace unrelated coffee preparation operations.
(define (domain kitchen)
  (:requirements :strips :equality)
  (:predicates (IsGripper ?gripper)
               (IsFood ?food)
               (IsPose ?object ?pose)
               (IsGrasp ?object ?grasp)
               (IsControl ?control)
               (IsSink ?surface)
               (IsStove ?surface)
               (AtPose ?object ?pose)
               (at-pose ?object ?pose)
               (AtSurface ?object ?surface)
               (Grasped ?object ?grasp)
               (HoldingConfig ?object ?grasp ?q)
               (Empty ?gripper)
               (Cleaned ?food)
               (Cooked ?food)
               (is-object ?object)
               (is-surface ?surface)
               (is-pose ?pose)
               (is-grasp ?object ?grasp)
               (grasp-pose ?object ?object-pose ?grasp ?pick-pose)
               (is-config ?q)
               (is-trajectory ?trajectory)
               (home-config ?q)
               (kinematic ?pose ?q)
               (placement ?object ?surface ?pose)
               (stack-placement ?object ?surface ?surface-pose ?pose)
               (motion ?q1 ?q2 ?trajectory)
               (collision-free ?q1 ?q2)
               (pick-plan ?object ?object-pose ?home ?grasp ?pick-pose
                          ?pick-q ?transit)
               (place-plan ?object ?surface ?pick-q ?home ?pose ?place-q
                           ?transfer ?return)
               (stack-plan ?object ?surface ?surface-pose ?pick-q ?home ?pose
                           ?place-q ?transfer ?return))

  (:action pick
    :parameters (?gripper ?food ?surface ?object-pose ?grasp ?pick-pose
                 ?home ?pick-q ?transit)
    :precondition (and (IsGripper ?gripper) (IsFood ?food)
                       (AtSurface ?food ?surface) (AtPose ?food ?object-pose)
                       (Empty ?gripper)
                       (pick-plan ?food ?object-pose ?home ?grasp ?pick-pose
                                  ?pick-q ?transit))
    :effect (and (Grasped ?food ?grasp) (HoldingConfig ?food ?grasp ?pick-q)
                 (not (AtSurface ?food ?surface))
                 (not (AtPose ?food ?object-pose))
                 (not (Empty ?gripper))))

  (:action place
    :parameters (?gripper ?food ?surface ?grasp ?pose ?pick-q ?home ?place-q
                 ?transfer ?return)
    :precondition (and (IsGripper ?gripper) (IsFood ?food)
                       (Grasped ?food ?grasp) (is-surface ?surface)
                       (HoldingConfig ?food ?grasp ?pick-q)
                       (place-plan ?food ?surface ?pick-q ?home ?pose ?place-q
                                   ?transfer ?return))
    :effect (and (AtSurface ?food ?surface) (AtPose ?food ?pose)
                 (Empty ?gripper)
                 (not (HoldingConfig ?food ?grasp ?pick-q))
                 (not (Grasped ?food ?grasp))))

  (:action clean
    :parameters (?food ?sink)
    :precondition (and (IsFood ?food) (IsSink ?sink)
                       (AtSurface ?food ?sink))
    :effect (Cleaned ?food))

  (:action cook
    :parameters (?food ?stove)
    :precondition (and (IsFood ?food) (IsStove ?stove)
                       (AtSurface ?food ?stove) (Cleaned ?food))
    :effect (Cooked ?food)))
