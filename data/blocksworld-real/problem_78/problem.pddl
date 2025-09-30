(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        blue - block
        orange - block
        purple - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable blue)
        (on orange blue)
        (on purple orange)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        ; red
        ; blue -> orange -> purple
        ; yellow
        ; green
    )
    (:goal (and (on blue purple)
        (on green blue)
        (on orange red)
        (on yellow orange)))
)
    