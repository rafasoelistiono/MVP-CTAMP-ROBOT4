; Adapted from caelan/pddlstream examples/blocksworld/domain.pddl.
; Canonical effects are unchanged. Extra parameters connect actions to streams.
(define (domain blocksworld)
  (:requirements :strips :equality)
  (:constants table)
  (:predicates (clear ?x)
               (on-table ?x)
               (arm-empty)
               (holding ?x)
               (on ?x ?y)
               (at-pose ?x ?pose)
               (AtPose ?x ?pose)
               (holding-config ?x ?grasp ?q)
               (is-object ?x)
               (is-surface ?x)
               (is-pose ?pose)
               (is-grasp ?x ?grasp)
               (grasp-pose ?x ?object-pose ?grasp ?pick-pose)
               (is-config ?q)
               (is-trajectory ?trajectory)
               (home-config ?q)
               (kinematic ?pose ?q)
               (placement ?x ?surface ?pose)
               (stack-placement ?x ?surface ?surface-pose ?pose)
               (motion ?q1 ?q2 ?trajectory)
               (collision-free ?q1 ?q2)
               (pick-plan ?x ?object-pose ?home ?grasp ?pick-pose ?pick-q ?transit)
               (place-plan ?x ?surface ?pick-q ?home ?pose ?place-q ?transfer ?return)
               (stack-plan ?x ?surface ?surface-pose ?pick-q ?home ?pose
                           ?place-q ?transfer ?return))

  (:action pickup
    :parameters (?ob ?source ?grasp ?pick-pose ?home ?pick-q ?transit)
    :precondition (and (clear ?ob) (on-table ?ob) (arm-empty)
                       (at-pose ?ob ?source)
                       (pick-plan ?ob ?source ?home ?grasp ?pick-pose
                                  ?pick-q ?transit))
    :effect (and (holding ?ob) (holding-config ?ob ?grasp ?pick-q)
                 (not (at-pose ?ob ?source))
                 (not (clear ?ob)) (not (on-table ?ob))
                 (not (arm-empty))))

  (:action putdown
    :parameters (?ob ?pose ?grasp ?pick-q ?home ?place-q ?transfer ?return)
    :precondition (and (holding ?ob) (home-config ?home)
                       (holding-config ?ob ?grasp ?pick-q)
                       (place-plan ?ob table ?pick-q ?home ?pose ?place-q
                                   ?transfer ?return))
    :effect (and (clear ?ob) (arm-empty) (on-table ?ob) (at-pose ?ob ?pose)
                 (not (holding-config ?ob ?grasp ?pick-q)) (not (holding ?ob))))

  (:action stack
    :parameters (?ob ?underob ?under-pose ?pose ?grasp ?pick-q ?home ?place-q
                 ?transfer ?return)
    :precondition (and (clear ?underob) (holding ?ob) (home-config ?home)
                       (holding-config ?ob ?grasp ?pick-q)
                       (at-pose ?underob ?under-pose)
                       (stack-plan ?ob ?underob ?under-pose ?pick-q ?home
                                   ?pose ?place-q ?transfer ?return))
    :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob) (at-pose ?ob ?pose)
                 (not (holding-config ?ob ?grasp ?pick-q))
                 (not (clear ?underob)) (not (holding ?ob))))

  (:action unstack
    :parameters (?ob ?underob ?source ?grasp ?pick-pose ?home ?pick-q ?transit)
    :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty)
                       (at-pose ?ob ?source)
                       (pick-plan ?ob ?source ?home ?grasp ?pick-pose
                                  ?pick-q ?transit))
    :effect (and (holding ?ob) (holding-config ?ob ?grasp ?pick-q)
                 (clear ?underob) (not (at-pose ?ob ?source))
                 (not (on ?ob ?underob)) (not (clear ?ob))
                 (not (arm-empty)))))
