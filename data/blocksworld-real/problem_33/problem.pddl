(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        red - block
        purple - block
        orange - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on red blue)
        (clear red)
        (ontable purple)
        (clear purple)
        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (clear green)
        ; blue -> red
        ; purple
        ; orange
        ; yellow
        ; green
    )
    (:goal (and (on green red)
        (on yellow green)
        (on purple yellow)
        (on orange purple)
        (on blue orange)))
)
    