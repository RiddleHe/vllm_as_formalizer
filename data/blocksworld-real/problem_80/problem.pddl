(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        orange - block
        purple - block
        yellow - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (clear blue)
        (ontable green)
        (clear green)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        ; blue
        ; green
        ; orange
        ; purple
        ; yellow
        ; red
    )
    (:goal (and (on yellow green)
        (on red blue)
        (on purple orange)))
)
    